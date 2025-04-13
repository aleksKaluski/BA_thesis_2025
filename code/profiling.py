from code.preprocessing import load_data as ld
from code.preprocessing import load_data as ld, merge_df as mf

# basic
import os

# case_specific
import spacy

# profling
from memory_profiler import profile

# change working directory
os.chdir(r"C:/BA_thesis/BA_v2_31.03")

print(f"working directory: {os.getcwd()}")
input_path = os.getcwd() + r'\files\corpus_data'
print(input_path)

@profile
def main():
    mf.merge_df('files/dfs')


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