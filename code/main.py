import os
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

import data_loader
import prompts
import api_client
import post_processor
import evaluator
import time
import json

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

def save_experiment_summary(args, stats, summary_path):
    summary_data = {
        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "task_type": args.task_type,
        "model_name": args.model_name,
        "dataset_seed": args.dataset_seed,
        "model_seed": args.model_seed,
        "temperature": args.temperature,
        "sampling_count": args.sampling_count if args.sampling_count else "Full",
        "input_file": args.input_file
    }

    summary_data.update(stats)
    summary_df = pd.DataFrame([summary_data])

    if not os.path.exists(summary_path):
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig', mode='w')
    else:
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig', mode='a', header=False)
        
    print(f"Saved Expeirmental Report.")


def main():
    parser = argparse.ArgumentParser(description="LLM Bias Evaluation Pipeline")
    
    # 1. 파일 입출력 관련
    parser.add_argument('--input_path', type=str, default='./data', help='input file path')
    parser.add_argument('--input_file', type=str, required=True, help='input file name')
    parser.add_argument('--output_path', type=str, default='./results', help='output file path')
    parser.add_argument('--output_file', type=str, default=None, help='output file name')
    
    # 2. 모델 및 실험 설정
    parser.add_argument('--model_name', type=str, default='gpt-5.1', help='model name')
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
    parser.add_argument('--dataset_seed', type=int, default=42, help='data seed')
    parser.add_argument('--model_seed', type=int, default=42, help='model seed')
    parser.add_argument('--sampling_count', type=int, default=10, help='the number of data per template')
    
    # 3. 태스크 타입
    parser.add_argument('--task_type', type=str, required=True, choices=['plot', 'character'], help='task type')
    parser.add_argument('--use_role', type=bool, default=False, help='Use task-specific persona')
    parser.add_argument('--run_type', type=str, default='test', required=False, choices=['full', 'test'])
    parser.add_argument('--inference_type', type=str, default='real-time', required=False, choices=['batch', 'real-time'])
    parser.add_argument('--max_token', type=int, default=512, required=False)
    args = parser.parse_args()


    # --- Preprocessing ---
    full_input_path = os.path.join(args.input_path, args.input_file)
    df = data_loader.load_and_preprocess(
        file_path=full_input_path,
        count=args.sampling_count,
        seed=args.dataset_seed,
        run_type=args.run_type
    )
    print("데이터 로드 완료: ", len(df))
    model_handler = api_client.get_model_handler(args.model_name)

    # ------------------------------------------------------------------
    # Batch API 실행
    # ------------------------------------------------------------------
    if args.inference_type == "batch":
        print(f":: Batch API Mode Detected (Model: {args.model_name}) ::")
        
        if not os.path.exists(args.output_path): os.makedirs(args.output_path)
        base_name = os.path.splitext(args.input_file)[0]
        
        jsonl_path = os.path.join(args.output_path, f"{base_name}_{args.task_type}_batch_request.jsonl")
        
        batch_lines = []
        metadata_dict = {}

        print(">> 배치 요청 파일 생성 중...")
        for index, row in tqdm(df.iterrows(), total=len(df)):
            prompt_text, metadata = prompts.construct_prompt(args.task_type, row, seed=args.dataset_seed)
            custom_id = f"req_{index}"
            
            use_role = getattr(args, 'use_role', False) 
            system_prompt = prompts.get_system_prompt(args.task_type, use_role=use_role)
            
            line = model_handler.prepare_batch_line(
                custom_id=custom_id,
                prompt=prompt_text,
                system_content=system_prompt,
                temperature=args.temperature,
                seed=args.model_seed
            )
            batch_lines.append(line)

            meta_row = row.to_dict()
            if metadata: meta_row.update(metadata)
            metadata_dict[custom_id] = meta_row

        # 파일 저장
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for line in batch_lines:
                f.write(line + '\n')
        
        batch_id = model_handler.submit_batch_job(jsonl_path)
        if not batch_id:
            print("배치 제출 실패. 프로그램을 종료합니다.")
            return

        print(f"배치 업로드 성공 (ID: {batch_id})\n")

        # ------------------------------------------------------------------
        # Polling Loop (배치 완료까지 대기)
        # ------------------------------------------------------------------
        output_file_id = None
        start_time = time.time()
        
        while True:
            time.sleep(5)
            
            batch_status = model_handler.check_batch_status(batch_id)
            if not batch_status:
                continue

            status = batch_status.status
            counts = batch_status.request_counts
            total = counts.total if counts else 0
            completed = counts.completed if counts else 0
            failed = counts.failed if counts else 0

            elapsed_seconds = int(time.time() - start_time)
            hours, rem = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

            print(f"\r >> Status: {status.upper()} | Completed: {completed}/{total} | Failed: {failed} | Time: {time_str}", end="")

            if status == "completed":
                print(f"\n배치 처리 완료. (총 소요시간: {time_str})")
                output_file_id = batch_status.output_file_id
                break
            elif status in ["failed", "expired", "cancelled"]:
                print(f"\n배치 작업 중단. (Status: {status})")
                return

        # ------------------------------------------------------------------
        # 결과 다운로드 및 병합 (Merge)
        # ------------------------------------------------------------------
        if output_file_id:
            print("결과 다운로드 중...")
            result_content = model_handler.retrieve_batch_result(output_file_id)
            results = []
            
            # JSONL 파싱
            for line in result_content.strip().split('\n'):
                if not line: continue
                res_json = json.loads(line)
                
                custom_id = res_json.get("custom_id")
                
                # 원본 메타데이터 찾기
                original_row = metadata_dict.get(custom_id)
                if not original_row:
                    continue

                # 응답 추출
                try:
                    raw_response = res_json['response']['body']['choices'][0]['message']['content'].strip()
                except:
                    raw_response = "Error"

                # postprocessing
                parsed_result, parsing_info = post_processor.parse_response(
                    task_type=args.task_type,
                    raw_response=raw_response,
                    metadata=original_row
                )

                # 최종 결과 행 구성
                final_row = original_row.copy()
                final_row['llm_raw_response'] = raw_response
                final_row['llm_parsed_result'] = parsed_result
                if parsing_info:
                    final_row.update(parsing_info)
                
                results.append(final_row)


    else:
        print(f":: Real-time Inferencing (Model: {args.model_name}) ::")
        base_name = os.path.splitext(args.input_file)[0]
        use_role = getattr(args, 'use_role', False) 
        system_prompt = prompts.get_system_prompt(args.task_type, use_role=use_role)
        
        results = []

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
            prompt_text, prompt_metadata = prompts.construct_prompt(args.task_type, row, seed=args.dataset_seed)
            
            raw_response = model_handler.generate(
                prompt=prompt_text,
                system_content=system_prompt,
                temperature=args.temperature,
                max_token=args.max_token,
                seed=args.model_seed
            )

            full_metadata = row.to_dict()
            if prompt_metadata:
                full_metadata.update(prompt_metadata)

            parsed_result, parsing_info = post_processor.parse_response(
                task_type=args.task_type,
                raw_response=raw_response,
                metadata=full_metadata
            )

            result_row = full_metadata.copy()
            result_row['llm_raw_response'] = raw_response
            result_row['llm_parsed_result'] = parsed_result
            if parsing_info: result_row.update(parsing_info)
            
            results.append(result_row)

        result_df = pd.DataFrame(results)


    # ------------------------------------------------------------------
    # 평가 및 저장 (Evaluation)
    # ------------------------------------------------------------------
    # raw 결과 저장
    raw_folder = os.path.join(args.output_path, "raw")
    os.makedirs(raw_folder, exist_ok=True)
    
    raw_save_name = f"{args.task_type}_{args.model_name}_d{args.dataset_seed}_m{args.model_seed}_raw.csv"
    raw_save_path = os.path.join(raw_folder, raw_save_name)
    result_df = pd.DataFrame(results)
    result_df.to_csv(raw_save_path, index=False, encoding='utf-8-sig')
    print(f"\nRaw 결과 저장 완료: {raw_save_path}")

    if 'custom_id' in result_df.columns:
        # req_0, req_1 에서 숫자만 추출하여 정렬
        result_df['sort_idx'] = result_df['custom_id'].apply(lambda x: int(x.split('_')[1]))
        result_df = result_df.sort_values('sort_idx').drop(columns=['sort_idx'])

    print("\nEvaluating...")
    evaluated_df, stats = evaluator.evaluate(result_df, args.task_type)   

    # evaluation 결과 저장
    eval_folder = os.path.join(args.output_path, "evaluation")
    os.makedirs(eval_folder, exist_ok=True)
    
    if args.output_file:
        save_name = args.output_file
    else:
        clean_model_name = args.model_name.replace("/", "_").replace("\\", "_")
        save_name = f"result_{args.task_type}_{clean_model_name}_d{args.dataset_seed}_m{args.model_seed}.csv"
    
    save_path = os.path.join(eval_folder, save_name)
    evaluated_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n최종 결과 저장 완료: {save_path}")

    # summary 저장
    current_date = datetime.now().strftime("%Y%m%d")
    summary_filename = f"summary_{args.task_type}_{args.model_name}_T{args.temperature}_{current_date}.csv"
    summary_file_path = os.path.join(args.output_path, summary_filename)
    save_experiment_summary(args, stats, summary_path=summary_file_path)
    return 

if __name__ == "__main__":
    main()