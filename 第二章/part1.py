#! java -mx6g -cp "stanford-corenlp-4.5.9/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 100000

from stanfordcorenlp import StanfordCoreNLP
import json

nlp = StanfordCoreNLP('http://localhost', port=9000)

sentence = "巨大的建筑，总是由一木一石叠起来的，我们何妨做做这一木一石呢?我时常做些零碎的事，就是为此。"

props = {
    'annotators': 'tokenize,ssplit,pos,depparse',
    'pipelineLanguage': 'zh',
    'ssplit.boundaryTokenRegex': '[。！？，；]',
    'tokenize.language': 'zh'
}

try:
    result = json.loads(nlp.annotate(sentence, properties=props))

    tokens = [word['word'] for sent in result['sentences'] for word in sent['tokens']]
    print("分词结果:", tokens)

    print("\n依存分析:")
    for sent in result['sentences']:
        for dep in sent['basicDependencies']:
            print(f"{dep['dependentGloss']} -> {dep['governorGloss']} ({dep['dep']})")
finally:
    nlp.close()
