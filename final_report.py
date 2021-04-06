import pandas as pd
from modules import final_results

if __name__ == "__main__":
    best_k = int(input("Enter the value of K: "))
    model_name = input("Enter the name of the model: ")
    title = input("Enter the title: ")

    final_results.get_res(best_k, model_name, title)