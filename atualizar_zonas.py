import re

# Lê o arquivo main.py
with open('main.py', 'r') as f:
    content = f.read()

# Novas zonas
novas_zonas = '''ZONES = {
    "collection": (0.00, 0.20, 0.30, 0.95),
    "sewing": (0.30, 0.20, 0.65, 0.95),
    "packaging": (0.65, 0.20, 1.00, 0.95),
    "needle_hazard": (0.42, 0.45, 0.52, 0.70),
}'''

# Substitui a seção ZONES
pattern = r'ZONES = \{.*?\}'
content = re.sub(pattern, novas_zonas, content, flags=re.DOTALL)

# Salva
with open('main.py', 'w') as f:
    f.write(content)

print("✅ Zonas atualizadas!")
print("\n📋 NOVAS ZONAS:")
print("  collection: (0.00, 0.20, 0.30, 0.95)")
print("  sewing:     (0.30, 0.20, 0.65, 0.95)")
print("  packaging:  (0.65, 0.20, 1.00, 0.95)")
print("  needle_hazard: (0.42, 0.45, 0.52, 0.70)")
print("\n🚀 Rode: python main.py")
