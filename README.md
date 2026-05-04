# Ročníkový projekt: Využitie veľkých jazykových modelov v programovaní

Tento projekt implementuje automatizovaný pipeline na optimalizáciu promptov pre LLM pri riešení programovacích úloh (Java). Systém vytiaha znalosti z odbornej literatúry, hodnotí ich užitočnosť pomocou kompilácie a testovania vygenerovaného kódu a následne využíva algoritmus dynamického programovania (Knapsack problem) na zostavenie optimálneho "Final-Promptu", ktorý sa zmestí do limitu tokenov.

## Štruktúra projektu

- `configs/` - Konfiguračné súbory a `.env` s API kľúčmi (nie je v gite).
- `dataset/` - Úlohy na testovanie (napr. HumanEval pre Javu).
- `lib/` - Zdrojová literatúra vo formáte PDF (napr. Effective Java).
- `processed/` - Vygenerované JSON dáta, predikcie a výsledky experimentov.
- `src/core/` - Hlavná logika (triedy ako `PDFParser`, `Predictor`, `Evaluator`, `KnapsackSolver`, `FinalPromptTester`).
- `src/runners/` - Spúšťacie skripty pre jednotlivé fázy pipeline.

## Inštalácia a spustenie

1. Nainštalujte potrebné knižnice:
   `pip install openai python-dotenv tiktoken pymupdf`

2. V zložke configs/ vytvorte súbor .env a pridajte svoj API kľúč:
   `OPENAI_API_KEY=vas_kluc_tu`

3. Spúšťajte skripty adresára runners/



## LiteLLm

Projekt je predpripravený na použitie s alternatívnymi modelmi pomocou knižnice `LiteLLM`

Ak chcete spustiť generovanie na vlastnom frameworku, vykonajte tieto kroky:

1. V priečinku `configs/` vytvorte alebo upravte súbor `settings.json`.
2. Definujte providera a konkrétny model:
```json
{
  "llm_config": {
    "provider": "openai",
    "model_name": "gpt-3.5-turbo"
  }
}
```
3. Otvorte triedu src/core/Predictor.py a v metóde get_llm_prediction() doplňte logiku pre volanie vášho lokálneho LLM v pripravenom bloku `elif self.provider == "litellm":`