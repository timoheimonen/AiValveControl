# AiValveControl
Älykäs venttiilin säätöjärjestelmä, joka hyödyntää PPO-mallia lämpötilan hallintaan. Tämä projekti simuloi ympäristöä ja käyttää vahvistusoppimista optimaalisen suorituskyvyn saavuttamiseksi.

## Lisenssi
Tämä projekti on lisensoitu MIT-lisenssillä. Katso lisätiedot [LICENSE](LICENSE) tiedostosta.

## Käyttö
python ./train.ai.py - kouluta malli, malli hyödyntää train_ai_data.py

python ./test_ai_gfx.py - graafinen pygame mallin toimivuudesta, malli hyödyntää train_ai_data.py

python ./test_ai.py - mallin testausta konsolissa.

Säädä mallin koulutusparametrejä train_ai.py olevassa PPO-osiossa.
Venttiilin viive, lämpötilat jne train_ai_data.py tiedostossa. Näitä käytetään koulutuksessa, sekä testauksessa.

## Ympäristö
Python versio 3.8.20
