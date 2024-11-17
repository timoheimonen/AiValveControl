# AiValveControl
Älykäs venttiilin säätöjärjestelmä, joka hyödyntää PPO-mallia lämpötilan hallintaan. Tämä projekti simuloi ympäristöä ja käyttää vahvistusoppimista optimaalisen suorituskyvyn saavuttamiseksi.

## Lisenssi
Tämä projekti on lisensoitu MIT-lisenssillä. Katso lisätiedot [LICENSE](LICENSE) tiedostosta.

## Käyttö
python ./train.ai.py - kouluta malli, malli hyödyntää train_ai_data.py
python ./test_ai_gfx.py - graafinen pygame mallin toimivuudesta, malli hyödyntää train_ai_data.py
python ./test_ai.py - mallin testausta kondsoliin.

Säädä mallin parametrejä train_ai.py olevassa PPO-osiossa.
Venttiilin viive, lämpötilat jne train_ai_data.py tiedostossa.

## Ympäristö
Python versio 3.8.20 Conda ympäristössä.
Condaan asennetut:
Name                    Version                   Build  Channel
absl-py                   2.1.0                    pypi_0    pypi
ale-py                    0.8.1                    pypi_0    pypi
blas                      1.0                    openblas  
box2d-py                  2.3.5                    pypi_0    pypi
ca-certificates           2024.9.24            hca03da5_0  
cachetools                5.5.0                    pypi_0    pypi
certifi                   2024.8.30                pypi_0    pypi
charset-normalizer        3.4.0                    pypi_0    pypi
cloudpickle               3.0.0            py38hca03da5_0  
contourpy                 1.1.1                    pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
etils                     1.3.0                    pypi_0    pypi
farama-notifications      0.0.4            py38hca03da5_0  
filelock                  3.16.1                   pypi_0    pypi
fonttools                 4.55.0                   pypi_0    pypi
fsspec                    2024.10.0                pypi_0    pypi
glfw                      2.7.0                    pypi_0    pypi
google-auth               2.36.0                   pypi_0    pypi
google-auth-oauthlib      1.0.0                    pypi_0    pypi
grpcio                    1.68.0                   pypi_0    pypi
gymnasium                 0.29.1                   pypi_0    pypi
gymnasium-notices         0.0.1                    pypi_0    pypi
idna                      3.10                     pypi_0    pypi
imageio                   2.35.1                   pypi_0    pypi
importlib-metadata        7.0.1            py38hca03da5_0  
importlib-resources       6.4.5                    pypi_0    pypi
importlib_metadata        7.0.1                hd3eb1b0_0  
jax-jumpy                 1.0.0            py38hca03da5_0  
jinja2                    3.1.4                    pypi_0    pypi
kiwisolver                1.4.7                    pypi_0    pypi
libcxx                    14.0.6               h848a8c0_0  
libffi                    3.4.4                hca03da5_1  
libgfortran               5.0.0           11_3_0_hca03da5_28  
libgfortran5              11.3.0              h009349e_28  
libopenblas               0.3.21               h269037a_0  
llvm-openmp               14.0.6               hc6e5704_0  
markdown                  3.7                      pypi_0    pypi
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.7.5                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
mujoco                    3.2.3                    pypi_0    pypi
ncurses                   6.4                  h313beb8_0  
networkx                  3.1                      pypi_0    pypi
numpy                     1.24.3           py38h1398885_0  
numpy-base                1.24.3           py38h90707a3_0  
oauthlib                  3.2.2                    pypi_0    pypi
openssl                   3.0.15               h80987f9_0  
packaging                 24.2                     pypi_0    pypi
pandas                    2.0.3                    pypi_0    pypi
pcre                      8.45                 hc377ac9_0  
pillow                    10.4.0                   pypi_0    pypi
pip                       24.2             py38hca03da5_0  
protobuf                  5.28.3                   pypi_0    pypi
pyasn1                    0.6.1                    pypi_0    pypi
pyasn1-modules            0.4.1                    pypi_0    pypi
pygame                    2.1.3                    pypi_0    pypi
pyopengl                  3.1.7                    pypi_0    pypi
pyparsing                 3.1.4                    pypi_0    pypi
python                    3.8.20               hb885b13_0  
python-dateutil           2.9.0.post0              pypi_0    pypi
pytz                      2024.2                   pypi_0    pypi
readline                  8.2                  h1a28f6b_0  
requests                  2.32.3                   pypi_0    pypi
requests-oauthlib         2.0.0                    pypi_0    pypi
rsa                       4.9                      pypi_0    pypi
setuptools                75.1.0           py38hca03da5_0  
shimmy                    0.2.1                    pypi_0    pypi
six                       1.16.0                   pypi_0    pypi
sqlite                    3.45.3               h80987f9_0  
stable-baselines3         2.3.2                    pypi_0    pypi
swig                      4.2.1.post0              pypi_0    pypi
sympy                     1.13.3                   pypi_0    pypi
tensorboard               2.14.0                   pypi_0    pypi
tensorboard-data-server   0.7.2                    pypi_0    pypi
tk                        8.6.14               h6ba3021_0  
torch                     2.4.1                    pypi_0    pypi
typing-extensions         4.11.0           py38hca03da5_0  
typing_extensions         4.11.0           py38hca03da5_0  
tzdata                    2024.2                   pypi_0    pypi
urllib3                   2.2.3                    pypi_0    pypi
werkzeug                  3.0.6                    pypi_0    pypi
wheel                     0.44.0           py38hca03da5_0  
xz                        5.4.6                h80987f9_1  
zipp                      3.20.2           py38hca03da5_0  
zlib                      1.2.13               h18a0788_1  

Pip asennetut:
Package                 Version
----------------------- -----------
absl-py                 2.1.0
ale-py                  0.8.1
box2d-py                2.3.5
cachetools              5.5.0
certifi                 2024.8.30
charset-normalizer      3.4.0
cloudpickle             3.0.0
contourpy               1.1.1
cycler                  0.12.1
etils                   1.3.0
Farama-Notifications    0.0.4
filelock                3.16.1
fonttools               4.55.0
fsspec                  2024.10.0
glfw                    2.7.0
google-auth             2.36.0
google-auth-oauthlib    1.0.0
grpcio                  1.68.0
gymnasium               0.29.1
gymnasium-notices       0.0.1
idna                    3.10
imageio                 2.35.1
importlib-metadata      7.0.1
importlib_resources     6.4.5
jax-jumpy               1.0.0
Jinja2                  3.1.4
kiwisolver              1.4.7
Markdown                3.7
MarkupSafe              2.1.5
matplotlib              3.7.5
mpmath                  1.3.0
mujoco                  3.2.3
networkx                3.1
numpy                   1.24.3
oauthlib                3.2.2
packaging               24.2
pandas                  2.0.3
pillow                  10.4.0
pip                     24.2
protobuf                5.28.3
pyasn1                  0.6.1
pyasn1_modules          0.4.1
pygame                  2.1.3
PyOpenGL                3.1.7
pyparsing               3.1.4
python-dateutil         2.9.0.post0
pytz                    2024.2
requests                2.32.3
requests-oauthlib       2.0.0
rsa                     4.9
setuptools              75.1.0
Shimmy                  0.2.1
six                     1.16.0
stable_baselines3       2.3.2
swig                    4.2.1.post0
sympy                   1.13.3
tensorboard             2.14.0
tensorboard-data-server 0.7.2
torch                   2.4.1
typing_extensions       4.11.0
tzdata                  2024.2
urllib3                 2.2.3
Werkzeug                3.0.6
wheel                   0.44.0
zipp                    3.20.2