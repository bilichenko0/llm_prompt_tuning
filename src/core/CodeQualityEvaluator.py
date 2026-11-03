import lizard
import math

class CodeQualityEvaluator:
    def evaluate(self, code_snippet: str) -> dict:
        
        default_metrics = {"ccn": 0.0, "nloc": 0, "mi": 0.0}
        
        if not code_snippet or not code_snippet.strip():
            return default_metrics

        try:
            analysis = lizard.analyze_file.analyze_source_code("Solution.java", code_snippet)
            
            if not analysis.function_list:
                return default_metrics
            
            total_ccn = sum(func.cyclomatic_complexity for func in analysis.function_list)
            avg_ccn = total_ccn / len(analysis.function_list)
            
            nloc = analysis.nloc
            
            
            log_nloc = math.log(nloc) if nloc > 0 else 0
            mi_raw = 171.0 - (5.2 * log_nloc) - (0.23 * avg_ccn) - (16.2 * log_nloc)
            mi = max(0.0, min(100.0, (mi_raw * 100.0) / 171.0))
            
            return {
                "ccn": round(avg_ccn, 2),
                "nloc": int(nloc),
                "mi": round(mi, 2)
            }
            
        except Exception as e:
            print(f"Lizard analysis failed: {e}")
            return default_metrics