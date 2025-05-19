import xml.etree.ElementTree as ET
import json
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def process_sentence(text):
    """Process sentence text with spaCy"""
    doc = nlp(text)
    tokens = [token.text for token in doc]
    pos = [token.pos_.upper() for token in doc]
    head = [token.head.i + 1 if token.head.i != i else 0 for i, token in enumerate(doc)]
    deprel = [token.dep_.lower() for token in doc]
    return tokens, pos, head, deprel


def xml_to_json(xml_string):
    root = ET.fromstring(xml_string)
    results = []

    for sentence in root.findall('sentence'):
        text = sentence.find('text').text
        tokens, pos, head, deprel = process_sentence(text)

        aspects = []
        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            for term in aspect_terms.findall('aspectTerm'):
                if term.attrib['polarity'] == 'conflict':
                    continue

                term_text = term.get('term')
                polarity = term.get('polarity')

                # Find token indices (simplified approach)
                term_tokens = term_text.split()
                term_from = None
                for i in range(len(tokens) - len(term_tokens) + 1):
                    if tokens[i:i + len(term_tokens)] == term_tokens:
                        term_from = i
                        break

                if term_from is not None:
                    aspects.append({
                        "term": term_tokens,
                        "from": term_from,
                        "to": term_from + len(term_tokens),
                        "polarity": polarity
                    })

        results.append({
            "token": tokens,
            "pos": pos,
            "head": head,
            "deprel": deprel,
            "aspects": aspects
        })

    return results


# XML data
xml_data = open(f'dataset/Restaurants_Test_Gold.xml', 'r', encoding='utf-8').read()

# Convert and save
json_data = xml_to_json(xml_data)

# Save to data.json
with open(f'dataset/test.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print("JSON data has been saved to data.json")

# XML data
xml_data = open(f'dataset/Restaurants_Train.xml', 'r', encoding='utf-8').read()

# Convert and save
json_data = xml_to_json(xml_data)

# Save to data.json
with open(f'dataset/train.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print("JSON data has been saved to data.json")