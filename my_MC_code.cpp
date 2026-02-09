// pi_mpi.cpp
#include <mpi.h>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long total_samples = 100000000; // 10^8
    long long samples_per_proc = total_samples / world_size;
    long long local_hits = 0;

    // Seed per processor to ensure randomness
    std::mt19937 gen(time(0) + world_rank);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (long long i = 0; i < samples_per_proc; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        if (x*x + y*y <= 1.0) local_hits++;
    }

    long long global_hits = 0;
    // Reduce all local hits to a global sum
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        double pi = 4.0 * (double)global_hits / (double)total_samples;
        std::cout << "MPI Estimate with " << world_size << " procs: " << pi << std::endl;
    }

    MPI_Finalize();
    return 0;
}