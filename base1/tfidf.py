import math
import spacy
import sys

nlp = spacy.load("pt_core_news_lg")

nome_arquivos = dict()

def gerarIndiceInvertido(texto, doc_id, indiceInvertido, caminhoArquivo):
    doc = nlp(texto.lower())

    tokens = [token.lemma_.lower() for token in doc if
              not token.is_stop and not token.is_space and not token.is_punct and not ' ' in token.lemma_]

    for token in tokens:
        if token in indiceInvertido:
            if doc_id in indiceInvertido[token]:
                indiceInvertido[token][doc_id] += 1
            else:
                indiceInvertido[token][doc_id] = 1
        else:
            indiceInvertido[token] = {doc_id: 1}

    nome_arquivos[doc_id] = caminhoArquivo

    return indiceInvertido

def gerarPonderacaoTFIDF(indiceInvertido, tfidf):
    # começando pelo idf
    idf = dict()
    N = len(nome_arquivos)

    for token, documentos in indiceInvertido.items():
        ni = len(documentos)
        idf[token] = math.log10(N / ni)

    # calculando o tfidf
    for token, documentos in indiceInvertido.items():
        for doc_id, frequencia in documentos.items():
            tf = 1 + math.log10(frequencia) if frequencia >= 1 else 0
            if doc_id not in tfidf:
                tfidf[doc_id] = {}
            tfidf[doc_id][token] = tf * idf[token]

    return tfidf

def processarDocumentos(caminhoBase):
    indiceInvertido = dict()
    tfidf = dict()

    with open(caminhoBase, "r", encoding="utf-8") as base:
        for doc_id, line in enumerate(base, start=1):
            caminhoArquivo = line.strip()
            with open(caminhoArquivo, "r", encoding="utf-8") as arquivo:
                texto = arquivo.read()
                gerarIndiceInvertido(texto, doc_id, indiceInvertido, caminhoArquivo)

    gerarPonderacaoTFIDF(indiceInvertido, tfidf)

    with open("indice.txt", "w", encoding="utf-8") as indice:
        for termo, documentos in indiceInvertido.items():
            lista_documentos = [f"{doc_id},{freq}" for doc_id, freq in documentos.items()]
            indice.write(f"{termo}: {' '.join(lista_documentos)}\n")

    with open("pesos.txt", "w", encoding="utf-8") as vetorDePesos:
        for doc_id, termos in tfidf.items():
            termos_pesos = [f"{termo}: {peso}" for termo, peso in termos.items() if peso > 0]
            if termos_pesos:
                nome_arquivo = nome_arquivos[doc_id]
                vetorDePesos.write(f"{nome_arquivo}: {' '.join(termos_pesos)}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    caminhoBase = sys.argv[1]
    processarDocumentos(caminhoBase)
