import requests, config

class LLM:
    def summarize(self, d):
        for fn in (self._gemini, self._groq, self._hf):
            text = fn(d)
            if text:
                return text
        return f"{d['retinopathy']} diabetic retinopathy detected. Consult a specialist."

    def _gemini(self, d):
        if not config.Config.GEMINI: return None
        try:
            from google import genai
            c = genai.Client(api_key=config.Config.GEMINI)
            r = c.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Explain {d['retinopathy']} diabetic retinopathy simply."
            )
            return r.candidates[0].content.parts[0].text
        except: return None

    def _groq(self, d):
        if not config.Config.GROQ: return None
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {config.Config.GROQ}"},
                json={"model": "llama3-8b-8192",
                      "messages":[{"role":"user","content":
                      f"Explain {d['retinopathy']} diabetic retinopathy simply."}]},
                timeout=8
            ).json()
            return r["choices"][0]["message"]["content"]
        except: return None

    def _hf(self, d):
        if not config.Config.HF: return None
        try:
            r = requests.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-base",
                headers={"Authorization": f"Bearer {config.Config.HF}"},
                json={"inputs": f"Explain {d['retinopathy']} diabetic retinopathy."},
                timeout=10
            ).json()
            return r[0]["generated_text"]
        except: return None
