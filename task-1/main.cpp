#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <mpi.h>
#include <omp.h>

#define MPI_MAIN_PROCESS_RANK (0)
#define MPI_TASK_TAG (0)
#define MPI_RESULT_TAG (1)

const double LEFT_BOUNDARY = 0.0;
const double RIGHT_BOUNDARY = 1.0;

typedef double TASK_ARR[3]; // [from, to] and dx
#define TASK_COUNT (3)
#define TASK_TYPE MPI_DOUBLE

inline MPI_Status receive_task(
    TASK_ARR task
) {
    MPI_Status status;
    
    MPI_Recv(
        task /* pointer to start */, 
        TASK_COUNT /* number of words */, 
        TASK_TYPE /* type of word */, 
        MPI_MAIN_PROCESS_RANK /* rank of sender */, 
        MPI_TASK_TAG /* tag */, 
        MPI_COMM_WORLD /* communicator: default - all world */,
        &status
    );

    return status;
}

inline MPI_Status receive_result_from(
    double* result,
    const int source
) {
    MPI_Status status;
    
    MPI_Recv(
        result /* pointer to start */, 
        1 /* number of words */, 
        MPI_DOUBLE /* type of word */, 
        source /* rank of sender */, 
        MPI_RESULT_TAG /* tag */, 
        MPI_COMM_WORLD /* communicator: default - all world */,
        &status
    );

    return status;
}

inline MPI_Status send_task_to(
    const TASK_ARR task, 
    const int dest
) {
    MPI_Status status;

    MPI_Send(
        task /* pointer to start */,
        TASK_COUNT /* number of words */,
        TASK_TYPE /* word type */, 
        dest /* rank of receiver */, 
        MPI_TASK_TAG /* tag */,
        MPI_COMM_WORLD /* communicator: default - all world */
    );

    return status;
}

inline MPI_Status send_result_to(
    const double result, 
    const int dest
) {
    MPI_Status status;

    MPI_Send(
        &result /* pointer to start */,
        1 /* number of words */,
        MPI_DOUBLE /* word type */, 
        dest /* rank of receiver */, 
        MPI_RESULT_TAG /* tag */,
        MPI_COMM_WORLD /* communicator: default - all world */
    );

    return status;
}

inline void generate_task_for(
    TASK_ARR task,
    const int N,
    const int p,
    const int dest
) {
    const double dx = (RIGHT_BOUNDARY - LEFT_BOUNDARY) / static_cast<double>(N);
    const int count = N / p + (p < N % p ? 1 : 0); 
    const double delta = count * dx;
    const double left = LEFT_BOUNDARY + delta * dest;
    const double right = left + delta;

    task[0] = left; task[1] = right; task[2] = dx;
}

inline double f(const double x) {
    const double inverse = 0.25L + x * x / 4.0L;
    return 1.0L / inverse;
}

inline double integrate(
    const TASK_ARR task,
    const int threads
) {
    const double left = task[0];
    const double right = task[1];
    const double dx = task[2];
    const int iterations = static_cast<int>((right - left) / dx);

    double* results = new double[threads];
    for (int i = 0; i < threads; ++i) results[i] = 0.0L;

    #pragma omp parallel num_threads(threads) 
    {
        #pragma omp for schedule(static, iterations / threads) nowait
        for (int i = 0; i < iterations; ++i) {
            const int thread_id = omp_get_thread_num();
            const double current = left + dx * i;
            const double leftf = f(current);
            const double rightf = f(current + dx);
            const double mean = (leftf + rightf) / 2.0L;
            results[thread_id] += mean * dx;
        }
    }
    
    double result = 0.0L;
    for (int i = 0; i < threads; ++i) result += results[i];
    delete[] results;

    return result;
}

void parse_arg(
    const char* name, 
    int* result,
    int argc, 
    char** argv
) {
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            if (i + 1 == argc) {
                std::cout << "WARNING: No arg for " << name 
                            << " - ignored" << std::endl;
                return;
            }
            int arg = std::atoi(argv[i + 1]);
            if (arg <= 0) {
                std::cout << "WARNING: Incorrect arg for " << name 
                            << " - ignored" << std::endl;
                return;
            }
            *result = arg;
        }
    }
}

int main(int argc, char** argv) {
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    
    MPI_Init(&argc, &argv);

    int N = 1000;
    int T = 4;

    parse_arg("-N", &N, argc, argv);
    parse_arg("-T", &T, argc, argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const double startt = MPI_Wtime();

    MPI_Status status;
    TASK_ARR task;
    if (world_rank == MPI_MAIN_PROCESS_RANK) {
        for (int id = 0; id < world_size; ++id) {
            if (id == MPI_MAIN_PROCESS_RANK) continue;

            generate_task_for(task, N, world_size, id);
            send_task_to(task, id);
        }

        generate_task_for(task, N, world_size, MPI_MAIN_PROCESS_RANK);
    } else receive_task(task);
    
    const double taskresult = integrate(task, T);
    double result = taskresult;

    if (world_rank == MPI_MAIN_PROCESS_RANK) {
        for (int id = 0; id < world_size; ++id) {
            if (id == MPI_MAIN_PROCESS_RANK) continue;

            double subresult;
            receive_result_from(&subresult, id);
            result += subresult;
        }
    } else send_result_to(result, MPI_MAIN_PROCESS_RANK);

    const double elapsedt = MPI_Wtime() - startt;

    std::cout << "Task result for " << world_rank << ": ["
                << task[0] << "; " << task[1] << "] by " 
                << task[2] << " = " << taskresult << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == MPI_MAIN_PROCESS_RANK) {
        std::cout << "Overall result = " << result << std::endl
                    << "Elapsed time - " << elapsedt << "s" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
