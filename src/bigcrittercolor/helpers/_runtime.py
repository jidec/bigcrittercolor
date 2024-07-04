import time

def _runtime(fun, *args, **kwargs):
    # Record the time before running the function
    start_time = time.time()

    # Execute the function with the provided arguments
    result = fun(*args, **kwargs)

    # Calculate the total time taken
    end_time = time.time()
    total_time = end_time - start_time

    print("Total time taken: {:.2f} seconds".format(total_time))

    return(total_time)