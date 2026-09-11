# Classificador de Exame — Câncer de Mama

Aplicação web que recebe o CSV de um exame e classifica o tumor como **maligno** ou **benigno**,
usando um modelo KNN treinado no dataset.

## Exploração

### Qual dataset foi escolhido e qual problema ele representa?

Breast Cancer Dataset <br>
https://www.kaggle.com/datasets/erdemtaha/cancer-data

### Qual é a variável-alvo (target) que será prevista?

A coluna diagnosis.


### Quais são as classes possíveis?

Duas, Maligno ou Benigno.

### Quais informações serão utilizadas como entrada do modelo?

Todas as colunas passadas no dataset.

### Quem utilizaria essa aplicação e com qual finalidade?

Seria utilizado por um patologista, para ter uma segunda opinião sobre o tumor.

### O que a aplicação fará com a classificação produzida pelo modelo?

Ela retorna o resultado dos dados passados, através do exame de PAAF.

### Como seria a interface ou experiência de uso dessa solução?

Inicialmente uma página única para passar os dados do exame e receber a validação do modelo.

## Como rodar

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python model.py
./venv/bin/uvicorn app:app --reload
```

Abra <http://127.0.0.1:8000>.

Para testar há dois arquivos prontos, extraídos do próprio dataset:
[exemplo_exame_maligno.csv](exemplo_exame_maligno.csv) e
[exemplo_exame_benigno.csv](exemplo_exame_benigno.csv).