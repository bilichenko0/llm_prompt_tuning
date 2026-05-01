class KnapsackSolver:
    def __init__(self, json_handler, token_limit: int, metric_name: str = "test_pass_rate"):
        self.jh = json_handler
        self.limit = token_limit
        self.metric_name = metric_name
    
    def solve(self):
        items = self.jh.get_by_status("evaluated")
        if not items:
            return []

        n = len(items)
        W = self.limit

        dp = [[0.0] * (W + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            item = items[i-1]
            weight = item["metadata"]["tokens"]["cl100k_base"]
            
            metrics = item.get("metadata", {}).get("metrics", {})
            value = metrics.get(self.metric_name) or 0.0

            for w in range(W + 1):
                if weight <= w:
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight] + value)
                else:
                    dp[i][w] = dp[i-1][w]

        selected_ids = []
        res_w = W
        for i in range(n, 0, -1):
            if dp[i][res_w] != dp[i-1][res_w]:
                item = items[i-1]
                selected_ids.append(item["id"])
                res_w -= item["metadata"]["tokens"]["cl100k_base"]

        return selected_ids