import cProfile
import pstats
import os

def do_cprofile(filename):
    """
    Decorator for function profiling.
    """
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            # Flag for do profiling or not.
            DO_PROF = os.getenv("PROFILING")
            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                # Sort stat by internal time.
                sortby = "tottime"
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result
        return profiled_func
    return wrapper

def print_profile(filename):
    """
    Print the profiling results.
    """
    profile = pstats.Stats(filename)
    profile.strip_dirs().sort_stats("cumulative").print_stats(20)
    profile.strip_dirs().sort_stats("time").print_stats(20)
    profile.strip_dirs().sort_stats("calls").print_stats(20)