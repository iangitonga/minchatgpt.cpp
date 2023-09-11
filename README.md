# minchatgpt.cpp
**minchatgpt.cpp** is a simple and minimal, pure-C++ implementation of a finetuned GPT-2 inference on CPU.
I finetuned the medium and large GPT-2 models to the Alpaca dataset to respond to questions and follow instructions.
My goal with this project is to recreate a baby version of ChatGPT by finetuning GPT-2 models to follow
instructions and answer questions and also provide a full implementation of the models inference on CPU. The
code used to finetune is scripts/train.ipynb.

**NOTE:** I am still in the process of developing this project and currently, the models are not able to
reach anywhere near ChatGPT or even finetuned LLaMa performance. My goal is to continue training the models
to have better performance and to also include the largest GPT-2 model with over one billion parameters.
I will also try to improve inference memory usage and performance.


## Install and Run minchatgpt.
```
git clone https://github.com/iangitonga/minchatgpt.cpp.git
cd minchatgpt.cpp/
g++ -std=c++17 -O3 -ffast-math minchatgpt.cpp -o minchatgpt
./minchatgpt

If you have an Intel CPU that supports AVX and f16c compile with the following
 command instead to achieve ~3x performance:
 
g++ -std=c++17 -O3 -ffast-math -mavx -mf16c minchatgpt.cpp -o minchatgpt
./minchatgpt
```
