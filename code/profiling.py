from code.preprocessing import load_data as ld

# basic
import os

# case_specific
import spacy

# profling
from memory_profiler import profile

# change working directory
os.chdir("/")

print(f"working directory: {os.getcwd()}")
input_path = os.getcwd() + r'\files\corpus_data'
print(input_path)

@profile
def main():
    nlp = spacy.load("en_core_web_sm")
    ld.load_data(dir_with_corpus_files=input_path,
                 nlp=nlp,
                 chunksize=40)


if __name__ == '__main__':
    import cProfile
    import pstats
    import io

    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    s = io.StringIO()
    sortby = 'cumtime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)
    print(s.getvalue())