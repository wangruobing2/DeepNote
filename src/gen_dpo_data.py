import random
import json
import os
import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vllm import LLM, SamplingParams
import argparse
from es_retrieve import retrieve
import yaml
import multiprocessing as mp
from utils import (
    p_template,
    seed_everything,
    LLM_score_gen_new_query,
    LLM_score_compare_note,
    LLM_score_init_note,
    LLM_score_rag,
)

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    choices=["2wikimultihopqa"],
    default="2wikimultihopqa",
    help="Name of the dataset to use",
)
parser.add_argument(
    "--input_data_path",
    type=str,
    default="../data/corpus/2wikimultihopqa/train.json",
    help="Path to the input data file in JSON/JSONL format",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="../data/dpo_data",
    help="Output directory path for generated data",
)
parser.add_argument(
    "--batch_size", type=int, default=9, help="Batch size for data processing"
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=15000,
    help="Number of samples to generate for each type.",
)
parser.add_argument(
    "--per_data_gen_num",
    type=int,
    default=9,
    help="The number of times to generate data for each sample.",
)
parser.add_argument(
    "--model",
    type=str,
    choices=["llama-3.1-8b-instruct", "qwen2.5-7b-instruct"],
    default="llama-3.1-8b-instruct",
    help="Base model used for data generation",
)
parser.add_argument(
    "--score_model",
    type=str,
    choices=["gpt-4o-mini-2024-07-18"],
    default="gpt-4o-mini-2024-07-18",
    help="Model used for scoring",
)
parser.add_argument(
    "--gpu_memory_utilization",
    type=float,
    default=0.9,
    help="GPU memory utilization limit",
)
parser.add_argument(
    "--device",
    type=str,
    default="0,1,2,3,4,5,6,7",
    help="Comma separated list of devices to use for parallel processing",
)
args = parser.parse_args()


def load_and_sample_data(file_path, sample_size):
    with open(file_path, "r", encoding="utf-8") as f:
        data = (
            [json.loads(line.strip()) for line in f]
            if "jsonl" in file_path
            else json.load(f)
        )
    return random.sample(data, sample_size)


def retrieve_q(question, top_k):
    refs = retrieve(args.dataset, question, topk=top_k)
    text = ""
    for ref in refs:
        text += f"Title: {ref['title']}\nText: {ref['paragraph_text']} \n\n"
    return text


class CustomDataset(Dataset):
    def __init__(self, data_list, args):
        self.data_list = data_list
        self.args = args

    def __getitem__(self, index):
        item = self.data_list[index]
        item["id"] = index
        return item

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        batch = [data for data in batch]
        if not batch:
            return None
        ids = [f["id"] for f in batch]
        questions = [f.get("question", None) for f in batch]
        answers = [f.get("answer", None) for f in batch]
        return {
            "ids": ids,
            "questions": questions,
            "answers": answers,
        }


def gen_data(split, device_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(max_tokens=1024, temperature=1, top_p=0.9)

    def call_llm(prompts, params=[sampling_params]):
        if "llama" in args.model:
            model_template = "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "qwen" in args.model or "minicpm" in args.model:
            model_template = (
                "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
        prompts = [model_template.format(prompt=p) for p in prompts]

        responses = [
            response.outputs[0].text for response in llm.generate(prompts, params)
        ]
        return responses

    temperature_list = [0.1, 0.5, 0.9]
    top_p_list = [0.1, 0.5, 0.9]
    param_combinations = [
        (temp, top_p) for temp in temperature_list for top_p in top_p_list
    ]
    param_combinations = (
        param_combinations[: args.per_data_gen_num]
        if args.per_data_gen_num <= len(param_combinations)
        else param_combinations
        + random.choices(
            param_combinations, k=args.per_data_gen_num - len(param_combinations)
        )
    )

    params_dict = {"max_tokens": 1024}

    for batch in tqdm(split):
        if not batch:
            continue

        batch.update({"prompts": []})
        (
            batch_temp_rag,
            batch_temp_gen_query,
            batch_temp_init_note,
            batch_temp_refine_note,
        ) = ([], [], [], [])
        # Generate rag
        for id, question in enumerate(batch["questions"]):
            top_k = random.randint(3, 9)
            refs = retrieve_q(question, top_k)
            rag_prompt = p_template("base", {"query": question, "refs": refs})
            init_note_prompt = p_template(
                "init_note", {"query": question, "refs": refs}
            )
            rag_sampling_params = [
                SamplingParams(**params_dict, temperature=temp, top_p=top_p)
                for temp, top_p in param_combinations
            ]
            rags_generated = call_llm(
                [rag_prompt] * len(rag_sampling_params), rag_sampling_params
            )

            best_rag, worst_rag = LLM_score_rag(rags_generated, batch["answers"][id])

            if best_rag and worst_rag:
                batch_temp_rag.append(
                    {
                        "id": f"rag-{batch['ids'][id]}",
                        "raw_question": question,
                        "prompt": rag_prompt,
                        "chosen": best_rag,
                        "rejected": worst_rag,
                        "data_type": "rag",
                        "gen_response_list": [
                            {
                                "index": idx,
                                "response": response,
                                "temperature": param_combination[0],
                                "top_p": param_combination[1],
                            }
                            for idx, (response, param_combination) in enumerate(
                                zip(rags_generated, param_combinations)
                            )
                        ],
                    }
                )

            # Generate init_note

            init_note_sampling_params = [
                SamplingParams(**params_dict, temperature=temp, top_p=top_p)
                for temp, top_p in param_combinations
            ]
            init_notes_generated = call_llm(
                [init_note_prompt] * len(init_note_sampling_params),
                init_note_sampling_params,
            )

            best_init_note, worst_init_note = LLM_score_init_note(
                args.score_model, question, refs, init_notes_generated
            )
            if best_init_note and worst_init_note:
                batch_temp_init_note.append(
                    {
                        "id": f"init_note-{batch['ids'][id]}",
                        "raw_question": question,
                        "prompt": init_note_prompt,
                        "chosen": best_init_note,
                        "rejected": worst_init_note,
                        "data_type": "init_note",
                        "gen_response_list": [
                            {
                                "index": idx,
                                "response": response,
                                "temperature": param_combination[0],
                                "top_p": param_combination[1],
                            }
                            for idx, (response, param_combination) in enumerate(
                                zip(init_notes_generated, param_combinations)
                            )
                        ],
                    }
                )

            # Generate queries
            gen_query_prompt = p_template(
                "gen_new_query",
                {"query": question, "note": best_init_note, "query_log": []},
            )
            gen_query_sampling_params = [
                SamplingParams(**params_dict, temperature=temp, top_p=top_p)
                for temp, top_p in param_combinations
            ]
            responses = call_llm(
                [gen_query_prompt] * len(gen_query_sampling_params),
                gen_query_sampling_params,
            )

            gen_querys = responses

            best_query, worst_query = LLM_score_gen_new_query(
                args.score_model, question, best_init_note, gen_querys, []
            )

            refs = retrieve_q((best_query + "\n" + question)[:500], top_k)
            refine_note_prompt = p_template(
                "refine_note", {"query": question, "refs": refs, "note": best_init_note}
            )
            batch["prompts"].append(refine_note_prompt)
            if best_query and worst_query:
                batch_temp_gen_query.append(
                    {
                        "id": f"new_query-{batch['ids'][id]}",
                        "raw_question": question,
                        "prompt": gen_query_prompt,
                        "chosen": best_query,
                        "rejected": worst_query,
                        "data_type": "gen_new_query",
                        "gen_response_list": [
                            {
                                "index": idx,
                                "response": gen_query,
                                "temperature": param_combination[0],
                                "top_p": param_combination[1],
                            }
                            for idx, (gen_query, param_combination) in enumerate(
                                zip(gen_querys, param_combinations)
                            )
                        ],
                    }
                )

        # Generate refined notes
        for temp, top_p in param_combinations:
            sampling_params = SamplingParams(
                **params_dict, temperature=temp, top_p=top_p
            )
            prompts = batch["prompts"]

            responses = [call_llm(prompt, temp, top_p) for prompt in prompts]

            for idx, response in enumerate(responses):
                if len(batch_temp_refine_note) <= idx:
                    batch_temp_refine_note.append(
                        {
                            "id": f"refine_note-{batch['ids'][idx]}",
                            "raw_question": batch["questions"][idx],
                            "prompt": prompts[idx],
                            "data_type": "refine_note",
                            "gen_response_list": [],
                        }
                    )
                score = LLM_score_compare_note(
                    args.score_model, batch["questions"][idx], best_init_note, response
                )
                batch_temp_refine_note[idx]["gen_response_list"].append(
                    {
                        "index": len(batch_temp_refine_note[idx]["gen_response_list"]),
                        "response": response,
                        "score_flag": score,
                        "temperature": temp,
                        "top_p": top_p,
                    }
                )

        for refine_note in batch_temp_refine_note:
            chosen_list = [
                item["response"]
                for item in refine_note["gen_response_list"]
                if item["score_flag"]
            ]
            rejected_list = [
                item["response"]
                for item in refine_note["gen_response_list"]
                if not item["score_flag"]
            ]
            refine_note["chosen"] = random.choice(chosen_list) if chosen_list else ""
            refine_note["rejected"] = (
                random.choice(rejected_list) if rejected_list else ""
            )
        batch_temp_refine_note = [
            refine_note
            for refine_note in batch_temp_refine_note
            if refine_note["chosen"] and refine_note["rejected"]
        ]
        batch_temp = (
            batch_temp_init_note
            + batch_temp_gen_query
            + batch_temp_refine_note
            + batch_temp_rag
        )
        if batch_temp:
            with open(output_file, "a", encoding="utf-8") as f:
                json_lines = [
                    json.dumps(data, ensure_ascii=False) for data in batch_temp
                ]
                f.write("\n".join(json_lines) + "\n")


def main():

    visible_devices = args.device.split(",")
    num_devices = len(visible_devices)

    data = load_and_sample_data(args.input_data_path, args.num_samples)

    split_data = lambda lst, n: [lst[i::n] for i in range(n)]

    dataset = CustomDataset(data, args)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn
    )

    splits = split_data(list(dataloader), num_devices)
    processes = []

    for rank, device_id in enumerate(visible_devices):
        split_dataloader = splits[rank]
        p = mp.Process(target=gen_data, args=(list(split_dataloader), device_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    current_date = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    seed_everything(66)
    model_path = config["model"][args.model]
    output_file = os.path.join(
        args.output_path,
        f"dpo_data-{args.model}-score_{args.score_model}_num-{args.num_samples}_para-{args.per_data_gen_num}-{current_date}.jsonl",
    )
    main()
