import pandas as pd
import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

def prefix_to_natural_language(prefix, timestamps=None):
    lines = []
    for i, event in enumerate(prefix):
        if timestamps:
            time_str = f"at {timestamps[i]}"
        else:
            time_str = f"Step {i+1}"
        lines.append(f"{time_str}, '{event}' occurred.")
    return "\n".join(lines)

def get_activity_statistics(df, case_col, activity_col, timestamp_col):
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.statistics.end_activities.log import get as end_activities_get
    from pm4py.statistics.start_activities.log import get as start_activities_get

    df = df.rename(columns={
        case_col: 'case:concept:name',
        activity_col: 'concept:name',
        timestamp_col: 'time:timestamp'
    })

    for col in ['case:concept:name', 'concept:name', 'time:timestamp']:
        if col not in df.columns:
            df[col] = pd.NA

    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
    event_log = log_converter.apply(df, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    end_activities = end_activities_get.get_end_activities(event_log)
    start_activities = start_activities_get.get_start_activities(event_log)
    return start_activities, end_activities

def make_prompt(case_id, prefix, prefix_t, candidate_activities, context_description="a patient is undergoing process"):
    abstraction = prefix_to_natural_language(prefix, prefix_t)
    candidate_str = ', '.join(candidate_activities)
    prompt = f"""
Context:
A {context_description} (case_id={case_id}).
The following events have occurred so far:
{abstraction}

Please predict the next most likely activity from the following candidates only:
[{candidate_str}]

For your answer, output the next activity and provide a short reason for your choice.
Format:
Activity: <one of candidates>
Reason: <short reason>
"""
    return prompt

def make_prompt_with_stats(case_id, prefix, prefix_t, candidate_activities,
                          start_activities, end_activities, context_description="a patient is undergoing process", topk=3):
    abstraction = prefix_to_natural_language(prefix, prefix_t)
    candidate_str = ', '.join(candidate_activities)
    top_start = sorted(start_activities.items(), key=lambda x: x[1], reverse=True)[:topk]
    top_end = sorted(end_activities.items(), key=lambda x: x[1], reverse=True)[:topk]
    start_desc = ', '.join([f"{act} ({cnt})" for act, cnt in top_start])
    end_desc = ', '.join([f"{act} ({cnt})" for act, cnt in top_end])
    prompt = f"""
Context:
A {context_description} (case_id={case_id}).

[Process-level statistics for reference]
- Most common starting activities: {start_desc}
- Most common ending activities: {end_desc}

The following events have occurred so far:
{abstraction}

Please predict the next most likely activity from the following candidates only:
[{candidate_str}]

For your answer, output the next activity and provide a short reason for your choice.
Format:
Activity: <one of candidates>
Reason: <short reason>
"""
    return prompt

def parse_llama_output(response_text):
    lines = [l.strip() for l in response_text.split('\n') if l.strip()]
    activity, reason = "", ""
    for line in lines:
        if line.lower().startswith("activity:"):
            activity = line.split(":",1)[1].strip()
        elif line.lower().startswith("reason:"):
            reason = line.split(":",1)[1].strip()
    if not activity and lines:
        activity = lines[0]
    if not reason:
        reason = "No explicit reason provided by LLM."
    return activity, reason

def hf_llm_batch(prompts, tokenizer, model, device, batch_size=4, max_new_tokens=64):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2)
        for out in outputs:
            response = tokenizer.decode(out, skip_special_tokens=True)
            results.append(response)
    return results

def LLM_mistral(
    input_csv, output_csv,
    case_col, activity_col, timestamp_col,
    context_description="a patient is undergoing process",
    aug=0.1, batch_size=4, gpu_id=0, model_name="mistralai/Mistral-7B-Instruct-v0.2",
    use_stats=False, topk=3
):
    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(input_csv)
    df = df.dropna(how='all')
    df = df.sort_values(by=[case_col, timestamp_col])
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    activity_set = sorted(df[activity_col].dropna().astype(str).unique().tolist())

    # pm4p
    if use_stats:
        start_acts, end_acts = get_activity_statistics(df, case_col, activity_col, timestamp_col)

    # Prompt generate
    prompts = []
    prompt_info = []
    for case_id, group in df.groupby(case_col):
        group = group.sort_values(timestamp_col).reset_index(drop=True)
        original_rows = group.to_dict('records')
        candidate_indices = list(range(len(original_rows) - 1))
        num_aug = max(1, int(len(candidate_indices) * aug))
        insert_indices = sorted(random.sample(candidate_indices, num_aug))
        for i in insert_indices:
            prefix = [str(r[activity_col]) for r in original_rows[:i+1]]
            prefix_t = [str(r[timestamp_col]) for r in original_rows[:i+1]]
            if use_stats:
                prompt = make_prompt_with_stats(
                    case_id, prefix, prefix_t, activity_set,
                    start_acts, end_acts, context_description, topk
                )
            else:
                prompt = make_prompt(
                    case_id, prefix, prefix_t, activity_set, context_description
                )
            prompts.append(prompt)
            prompt_info.append((case_id, i, original_rows))

    # LLM inference
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()
    llm_responses = hf_llm_batch(prompts, tokenizer, model, device, batch_size=batch_size)

    # augmented row
    augmented_rows = []
    response_idx = 0
    for case_id, group in df.groupby(case_col):
        group = group.sort_values(timestamp_col).reset_index(drop=True)
        original_rows = group.to_dict('records')
        candidate_indices = list(range(len(original_rows) - 1))
        num_aug = max(1, int(len(candidate_indices) * aug))
        insert_indices = sorted(random.sample(candidate_indices, num_aug))
        insert_set = set(insert_indices)
        for i, row in enumerate(original_rows):
            augmented_rows.append({
                **row,
                "Aug": 0,
                "semantic_reason": "",
                "augmented_activity": "",
                "prompt": ""
            })
            if i in insert_set:
                response_text = llm_responses[response_idx]
                llm_act, reason = parse_llama_output(response_text)
                t_cur = row[timestamp_col]
                t_next = original_rows[i+1][timestamp_col]
                t_new = t_cur + (t_next - t_cur)/2
                aug_row = {
                    case_col: case_id,
                    activity_col: llm_act,
                    "resource": "",
                    timestamp_col: t_new,
                    "Aug": 1,
                    "semantic_reason": reason,
                    "augmented_activity": llm_act,
                    "prompt": prompts[response_idx].strip()
                }
                augmented_rows.append(aug_row)
                response_idx += 1

    result_df = pd.DataFrame(augmented_rows)
    result_df = result_df.sort_values(by=[case_col, timestamp_col]).reset_index(drop=True)
    result_df[timestamp_col] = result_df[timestamp_col].astype(str)
    result_df.to_csv(output_csv, index=False)

    num_original = (result_df['Aug'] == 0).sum()
    num_augmented = (result_df['Aug'] == 1).sum()
    augmentation_rate = num_augmented / num_original
    print(f"\n[증강 통계]")
    print(f"원본 activity 수: {num_original}")
    print(f"증강 activity 수: {num_augmented}")
    print(f"증강 비율: {augmentation_rate:.2%}")

    stat_df = pd.DataFrame([{
    "num_original": num_original,
    "num_augmented": num_augmented,
    "augmentation_rate": augmentation_rate
    }])

    # 확장자 앞에 '_stat'를 붙여 저장
    base, ext = os.path.splitext(output_csv)
    stat_csv = f"{base}_stat{ext}"

    stat_df.to_csv(f"{stat_csv}", index=False)



# ---- 사용 예시 ----
if __name__ == "__main__":
    # Credit.csv의 경우
    # LLM_mistral(
    #     input_csv="Credit.csv",
    #     output_csv="Credit_prompt_1_augmented_semantic_0.1_hf.csv",
    #     case_col="Case", activity_col="Activity", timestamp_col="Timestamp",
    #     context_description="This dataset records the sequence of activities for each customer using the bank’s credit service, including financial transactions and account management events.",
    #     aug=0.1, batch_size=8, gpu_id=0, use_stats=False
    # )

    # LLM_mistral(
    #     input_csv="Credit.csv",
    #     output_csv="Credit_prompt_1_augmented_semantic_0.3_hf.csv",
    #     case_col="Case", activity_col="Activity", timestamp_col="Timestamp",
    #     context_description="This dataset records the sequence of activities for each customer using the bank’s credit service, including financial transactions and account management events.",
    #     aug=0.3, batch_size=8, gpu_id=0, use_stats=False
    # )

    # LLM_mistral(
    #     input_csv="Credit.csv",
    #     output_csv="Credit_prompt_2_augmented_semantic_0.1_hf.csv",
    #     case_col="Case", activity_col="Activity", timestamp_col="Timestamp",
    #     context_description="This dataset records the sequence of activities for each customer using the bank’s credit service, including financial transactions and account management events.",
    #     aug=0.1, batch_size=8, gpu_id=1, use_stats=True
    # )

    # LLM_mistral(
    #     input_csv="Credit.csv",
    #     output_csv="Credit_prompt_2_augmented_semantic_0.3_hf.csv",
    #     case_col="Case", activity_col="Activity", timestamp_col="Timestamp",
    #     context_description="This dataset records the sequence of activities for each customer using the bank’s credit service, including financial transactions and account management events.",
    #     aug=0.3, batch_size=8, gpu_id=0.1, use_stats=True
    # )




    # LLM_mistral(
    #     input_csv="sepsis.csv",
    #     output_csv="sepsis_prompt_1_augmented_semantic_0.1_hf.csv",
    #     case_col="case_id", activity_col="activity", timestamp_col="timestamp",
    #     context_description="A patient is undergoing sepsis management in a hospital. \
    # Each patient (case) record consists of a sequence of medical events and activities \
    #     (e.g., laboratory tests, medication, ICU transfers, discharge) recorded with precise timestamps. \
    #         The data captures the order and timing of critical interventions and clinical observations, \
    #             reflecting real-world patient care processes for suspected or confirmed sepsis. \
    #                 The goal is to predict the next likely activity in the patient’s timeline, given the current sequence of events, to support early intervention and clinical decision-making.",
    #     aug=0.1, batch_size=8, gpu_id=0, use_stats=False
    # )

    # LLM_mistral(
    #     input_csv="sepsis.csv",
    #     output_csv="sepsis_prompt_1_augmented_semantic_0.3_hf.csv",
    #     case_col="case_id", activity_col="activity", timestamp_col="timestamp",
    #     context_description="A patient is undergoing sepsis management in a hospital. \
    # Each patient (case) record consists of a sequence of medical events and activities \
    #     (e.g., laboratory tests, medication, ICU transfers, discharge) recorded with precise timestamps. \
    #         The data captures the order and timing of critical interventions and clinical observations, \
    #             reflecting real-world patient care processes for suspected or confirmed sepsis. \
    #                 The goal is to predict the next likely activity in the patient’s timeline, given the current sequence of events, to support early intervention and clinical decision-making.",
    #     aug=0.3, batch_size=8, gpu_id=0, use_stats=False
    # )

    # LLM_mistral(
    #     input_csv="sepsis.csv",
    #     output_csv="sepsis_prompt_2_augmented_semantic_0.1_hf.csv",
    #     case_col="case_id", activity_col="activity", timestamp_col="timestamp",
    #     context_description="A patient is undergoing sepsis management in a hospital. \
    # Each patient (case) record consists of a sequence of medical events and activities \
    #     (e.g., laboratory tests, medication, ICU transfers, discharge) recorded with precise timestamps. \
    #         The data captures the order and timing of critical interventions and clinical observations, \
    #             reflecting real-world patient care processes for suspected or confirmed sepsis. \
    #                 The goal is to predict the next likely activity in the patient’s timeline, given the current sequence of events, to support early intervention and clinical decision-making.",
    #     aug=0.1, batch_size=8, gpu_id=0, use_stats=True
    # )

    # LLM_mistral(
    #     input_csv="sepsis.csv",
    #     output_csv="sepsis_prompt_2_augmented_semantic_0.3_hf.csv",
    #     case_col="case_id", activity_col="activity", timestamp_col="timestamp",
    #     context_description="A patient is undergoing sepsis management in a hospital. \
    # Each patient (case) record consists of a sequence of medical events and activities \
    #     (e.g., laboratory tests, medication, ICU transfers, discharge) recorded with precise timestamps. \
    #         The data captures the order and timing of critical interventions and clinical observations, \
    #             reflecting real-world patient care processes for suspected or confirmed sepsis. \
    #                 The goal is to predict the next likely activity in the patient’s timeline, given the current sequence of events, to support early intervention and clinical decision-making.",
    #     aug=0.3, batch_size=8, gpu_id=0, use_stats=True
    # )

    # LLM_mistral(
    #     input_csv="BPIC15_1.csv",
    #     output_csv="BPIC15_prompt_1_augmented_semantic_0.1_hf.csv",
    #     case_col="Case", activity_col="Activity", timestamp_col="Timestamp",
    #     context_description="This data records the business process of handling building permit \
    #         applications in a Dutch municipality, as captured in the BPI Challenge 2015 dataset. \
    #             Each case represents a single permit application, tracked from submission through various \
    #                 administrative and technical processing steps, up to final approval or rejection. For each application (case), \
    #                     the dataset logs the sequence and timing of key activities—such as document receipt, assessment, communication with stakeholders, decisions, and archiving. Your task is to predict the next likely activity in the process, \
    #                         given the applications event history up to now, following realistic administrative workflows.",
    #     aug=0.1, batch_size=8, gpu_id=0, use_stats=False
    # )

    # LLM_mistral(
    #     input_csv="BPIC15_1.csv",
    #     output_csv="BPIC15_prompt_1_augmented_semantic_0.3_hf.csv",
    #     case_col="Case", activity_col="Activity", timestamp_col="Timestamp",
    #     context_description="This data records the business process of handling building permit \
    #         applications in a Dutch municipality, as captured in the BPI Challenge 2015 dataset. \
    #             Each case represents a single permit application, tracked from submission through various \
    #                 administrative and technical processing steps, up to final approval or rejection. For each application (case), \
    #                     the dataset logs the sequence and timing of key activities—such as document receipt, assessment, communication with stakeholders, decisions, and archiving. Your task is to predict the next likely activity in the process, \
    #                         given the applications event history up to now, following realistic administrative workflows.",
    #     aug=0.3, batch_size=8, gpu_id=0, use_stats=False
    # )

    LLM_mistral(
        input_csv="BPIC15_1.csv",
        output_csv="BPIC15_prompt_2_augmented_semantic_0.1_hf.csv",
        case_col="Case", activity_col="Activity", timestamp_col="Timestamp",
        context_description="This data records the business process of handling building permit \
            applications in a Dutch municipality, as captured in the BPI Challenge 2015 dataset. \
                Each case represents a single permit application, tracked from submission through various \
                    administrative and technical processing steps, up to final approval or rejection. For each application (case), \
                        the dataset logs the sequence and timing of key activities—such as document receipt, assessment, communication with stakeholders, decisions, and archiving. Your task is to predict the next likely activity in the process, \
                            given the applications event history up to now, following realistic administrative workflows.",
        aug=0.1, batch_size=8, gpu_id=1, use_stats=True
    )

    LLM_mistral(
        input_csv="BPIC15_1.csv",
        output_csv="BPIC15_prompt_2_augmented_semantic_0.3_hf.csv",
        case_col="Case", activity_col="Activity", timestamp_col="Timestamp",
        context_description="This data records the business process of handling building permit \
            applications in a Dutch municipality, as captured in the BPI Challenge 2015 dataset. \
                Each case represents a single permit application, tracked from submission through various \
                    administrative and technical processing steps, up to final approval or rejection. For each application (case), \
                        the dataset logs the sequence and timing of key activities—such as document receipt, assessment, communication with stakeholders, decisions, and archiving. Your task is to predict the next likely activity in the process, \
                            given the applications event history up to now, following realistic administrative workflows.",
        aug=0.3, batch_size=8, gpu_id=1, use_stats=True
    )