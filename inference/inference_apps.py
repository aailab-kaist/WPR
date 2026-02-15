from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from dschat.utils.data.raw_datasets import OpenAISummarizeDataset, DahoasFullhhrlhfDataset, OpenaiWebgptcomparisonsDataset, CodeparrotAPPSDataset
from dschat.utils.model.model_utils import create_critic_model
from tqdm import tqdm
import os
import json
import torch
import multiprocessing
from testing_util import run_test
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0]

def generate_response(model, tokenizer, temperature, inputs):
    generate_kwargs = {
        "temperature": temperature,
        "do_sample": True if temperature > 0 else False,
        "top_k": 5
    }
    generated = model.generate(**inputs, max_new_tokens=512, **generate_kwargs)
    outputs = [tokenizer.decode(o[inputs["input_ids"].size(1):], skip_special_tokens=True) for o in generated]
    return outputs

def _apps_reward_from_status(status):
    """
    status: [-2] compile error, [-1] runtime error, [True/False,...] 테스트 결과
    """
    def to_bool(x):
        if isinstance(x, np.ndarray):
            x = x.item()
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        return x

    status = list(map(to_bool, status))
    if -2 in status:
        return -1.0      # compile error
    if -1 in status:
        return -0.6      # runtime error
    n_pass = sum(s is True for s in status)
    n_fail = sum(s is False for s in status)
    frac = (n_pass / (n_pass + n_fail)) if (n_pass + n_fail) > 0 else 0.0
    return -0.3 + 1.3 * frac


def get_reward_score(prompts, responses, metas):
    assert len(prompts) == len(responses)
    rewards = []

    for i, (prompt, code, meta) in enumerate(zip(prompts, responses, metas)):
        apps_sample = meta
        try:
            status = check_correctness(apps_sample, code, timeout=10, debug=False)
        except Exception:
            status = [-2]

        rewards.append(_apps_reward_from_status(status))

    return rewards

def main(args):
    PROJ_PATH=args.proj_path

    if args.dataset == "summarize":
        raw_dataset = OpenAISummarizeDataset("", 1234, 0, "openai/summarize_from_feedback")
    elif args.dataset == "hh-rlhf":
        raw_dataset = DahoasFullhhrlhfDataset("", 1234, 0, "Dahoas/full-hh-rlhf")
    elif args.dataset == "webgpt":
        raw_dataset = OpenaiWebgptcomparisonsDataset(".", 1234, 0, "openai/webgpt_comparisons")
    elif args.dataset == "apps":
        raw_dataset = CodeparrotAPPSDataset(".", 1234, 0, "codeparrot/apps")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    validation = raw_dataset.get_eval_data()
    if args.dataset not in ["webgpt", "apps"]:
        validation = validation.shuffle(seed=1234)
    prompts, chosens, metas = [], [], []
    for sample in validation:
        prompts.append(raw_dataset.get_prompt(sample))
        chosen = raw_dataset.get_chosen(sample)
        chosens.append(chosen)
        metas.append(raw_dataset.get_meta(sample))
    num_samples = args.num_samples
    prompts = prompts[:num_samples]
    chosens = chosens[:num_samples]
    metas = metas[:num_samples]
    batch_size = args.batch_size
    torch.manual_seed(1234)
    step = args.model.split('/')[-1]
    model_name = args.model.split('/')[-2]
    with torch.no_grad():
        if args.no_actor_folder:
            tokenizer = AutoTokenizer.from_pretrained(f"{args.model}", device_map=f'cuda:{args.gpu}')
            model = AutoModelForCausalLM.from_pretrained(f"{args.model}", device_map=f"cuda:{args.gpu}", torch_dtype=torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(f"{args.model}/actor", device_map=f'cuda:{args.gpu}')
            model = AutoModelForCausalLM.from_pretrained(f"{args.model}/actor", device_map=f"cuda:{args.gpu}", torch_dtype=torch.float16)
        inferences = []
        model.eval()
        for i in tqdm(range(0, len(prompts), batch_size)):
            inputs = tokenizer.batch_encode_plus(prompts[i: i + batch_size], return_tensors="pt", truncation=True, max_length=1024, padding=True).to(model.device)

            if args.ref_scores:
                chosen_reward_score = get_reward_score(prompts[i: i + batch_size], chosens[i: i + batch_size], metas[i: i + batch_size])

                for j in range(len(chosen_reward_score)):
                    inferences.append({"prompt": prompts[i + j], "response": chosens[i + j], "reward": chosen_reward_score[j]})
            else:
                response = generate_response(model, tokenizer, args.temperature, inputs)
                reward_score = get_reward_score(prompts[i: i + batch_size], response, metas[i: i + batch_size])

                for j in range(len(response)):
                    inferences.append({"prompt": prompts[i + j], "response": response[j], "reward": reward_score[j]})

    # os.makedirs(f"{PROJ_PATH}/results/{args.dataset}/temperature={args.temperature}/{model_name}_{step}", exist_ok=True)

    if args.ref_scores:
        with open(f'{PROJ_PATH}/results/{args.dataset}/temperature={args.temperature}/dataset_{num_samples}.jsonl', 'w') as f:
            for inference in inferences:
                f.write(json.dumps(inference) + '\n')
    else:
        with open(f'{PROJ_PATH}/results/{args.dataset}/temperature={args.temperature}/{model_name}_{step}_{num_samples}.jsonl', 'w') as f:
            for inference in inferences:
                f.write(json.dumps(inference) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--reward-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="summarize")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ref-scores", action='store_true')
    parser.add_argument("--no-actor-folder", action='store_true')
    parser.add_argument("--num-samples", type=int, default=2000)
    args = parser.parse_args()
    main(args)