import argparse
from llama_cpp import Llama
from gemma import Gemma3ChatHandler


def main(args):
    llm = Llama(model_path=args.model, n_ctx=2 ** 15, n_gpu_layers=100)
    h = Gemma3ChatHandler(args.mmproj)
    h.verbose = True
    # messages = [
    #      {
    #          "role": "system",
    #          "content": [{"type": "text", "text": "You are a helpful assistant."}]
    #      },
    #      {
    #          "role": "user",
    #          "content": [
    #              {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
    #              {"type": "text", "text": "What animal is on the candy in the following image? https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"}
    #          ]
    #      }
    # ]
    messages = [
         {
             "role": "system",
             "content": [{"type": "text", "text": "You are a helpful assistant."}]
         },
         {
             "role": "user",
             "content": [
                 {"type": "image", "url": "file:/home/joe/nn_mat_test_2.png"},
                 {"type": "text", "text": "Can you extract text from this image and describe what the extracted text means?. You MUST USE LaTeX for all math file:/home/joe/nn_mat_test_2.png"}
             ]
         }
    ]
    stream = h(llama=llm, messages=messages, stream=True)
    for output in stream:
        part = output["choices"][0]["delta"]
        if "content" in part:
            print(part["content"], end="")
    print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("--mmproj")
    args = parser.parse_args()
    main(args)
