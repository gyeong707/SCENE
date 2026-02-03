import re
import ast

def clean_text(text):
    """특수문자 제거 및 공백 정규화"""
    if not text: return ""
    text = re.sub(r"[\[\]\'\"\.]", "", str(text))
    return re.sub(r"\s+", " ", text).strip()

def parse_response(task_type, raw_response, metadata=None):
    # 0. 입력값 예외 처리
    if not raw_response or raw_response == "Error":
        return "Error", {"parsing_failed": True, "reason": "empty_or_error_input"}
    
    # ==========================================================================
    # 1. Character Task
    # ==========================================================================
    if task_type == 'character':
        
        # 1-1. 불필요한 태그 제거 (<think>, assistantfinal 등)
        if '</think>' in raw_response:
            raw_response = raw_response.split('</think>')[-1].strip()
        
        if 'assistantfinal' in raw_response:
            raw_response = raw_response.split('assistantfinal')[-1].strip()

        val_a, val_b = "", ""
        # print("Raw response: ", raw_response) 
        
        # 1-2. 줄 단위로 A, B 답변 추출
        for line in raw_response.split('\n'):
            line = line.strip()
            if re.search(r'^A\s*[:\.]', line, re.IGNORECASE):
                val_a = clean_text(re.sub(r'^A\s*[:\.]', '', line, flags=re.IGNORECASE))
            elif re.search(r'^B\s*[:\.]', line, re.IGNORECASE):
                val_b = clean_text(re.sub(r'^B\s*[:\.]', '', line, flags=re.IGNORECASE))
        
        # 1-3. 포맷 매칭 실패 (A나 B가 없음) -> "Error" 반환
        if not val_a or not val_b:
            print(f"[Validation Error] Reason: format_mismatch_missing_AB | A: '{val_a}', B: '{val_b}'")
            return "Error", {
                "parsing_failed": True, 
                "reason": "format_mismatch_missing_AB", 
                "raw": raw_response
            }

        # 1-4. 메타데이터 기반 유효성 검증
        if metadata:
            entity_n1 = clean_text(metadata.get('N1_entity'))
            entity_n2 = clean_text(metadata.get('N2_entity'))
            valid_set = {entity_n1, entity_n2}
            valid_set.discard("") # 빈 문자열 제거
            
            # --- 검증 내부 함수 ---
            def validate_choice(extracted_text, candidates):
                found_matches = [c for c in candidates if c in extracted_text]
                
                if len(found_matches) > 1:
                    print(f"[Validation Error] Reason: ambiguous_multiple_matches | A: '{val_a}', B: '{val_b}'")
                    return None, "ambiguous_multiple_matches"
                
                if not found_matches:
                    print(f"[Validation Error] Reason: no_match_found | A: '{val_a}', B: '{val_b}'")
                    return None, "no_match_found"
                
                match = found_matches[0]
                if (len(extracted_text) - len(match)) > 5:
                    print(f"[Validation Error] excessive_noise(len_diff={len(extracted_text) - len(match)})")
                    return None, f"excessive_noise(len_diff={len(extracted_text) - len(match)})"

                return match, None
            # ---------------------

            final_a, err_a = validate_choice(val_a, valid_set)
            final_b, err_b = validate_choice(val_b, valid_set)
            
            if err_a or err_b:
                error_reason = err_a if err_a else err_b
                print(f"[Validation Error] Reason: {error_reason} | A: '{val_a}', B: '{val_b}'")
                
                return "Error", {
                    "parsing_failed": True, 
                    "reason": "validation_mismatch",
                    "detail": error_reason,
                    "expected": list(valid_set),
                    "got_raw": f"{val_a} / {val_b}"
                }
            
            val_a = final_a
            val_b = final_b

        parsed = f"{val_a}/{val_b}"
        # print("Parsed results: ", parsed)
        return parsed, None

    # ==========================================================================
    # 2. Plot Task
    # ==========================================================================
    elif task_type == 'plot':
        cleaned_response = raw_response
        
        if '</think>' in cleaned_response:
            cleaned_response = cleaned_response.split('</think>')[-1].strip()
        
        if 'assistantfinal' in cleaned_response:
            cleaned_response = cleaned_response.split('assistantfinal')[-1].strip()
        
        output_match = re.search(r'<output>(.*?)</output>', cleaned_response, re.DOTALL)
        if output_match:
            cleaned_response = output_match.group(1).strip()
        
        match = re.search(r'\b([123])\b', cleaned_response)
        # print(cleaned_response)
        
        # 메타데이터에서 정답 매핑 로드
        answer_map = metadata.get('answer_map_obj')
        
        # 문자열로 저장된 경우 dict로 변환
        if isinstance(answer_map, str):
            try:
                answer_map = ast.literal_eval(answer_map)
            except:
                pass
                
        if match and answer_map:
            selected_num = match.group(1)  # 1, 2, 3만 추출
            
            # 매핑된 라벨 반환 (Biased, Counter, Neutral)
            parsed_type = answer_map.get(selected_num, "OutOfRange")
            # print(f"Selected: {selected_num} -> {parsed_type}")
            
            return parsed_type, {"selected_num": selected_num}
        else:
            return "ParsingError", {"parsing_failed": True, "cleaned_response": cleaned_response[:200]}

    return None, None