from google import genai

client = genai.Client()

for m in client.models.list():
    print(m.name, m.supported_methods)
