#ifndef SMARTSIM_CTEST_INT32_UTILS_H
#define SMARTSIM_CTEST_INT32_UTILS_H
#include <stdio.h>
#include <stdlib.h>

void test_result(int result, char *test){
    if (result) {
        fprintf(stdout, "SUCCESS: %c", *test);
        return;
    }
    else {
        fprintf(stderr, "FAILED: %c", *test);
        exit(-1);
    }
}

uint safe_rand(){
    uint random = (rand() % 254) + 1;
    return random;
}

float rand_float(){
    float random = ((float) safe_rand())/safe_rand();
    return random;
}

double rand_double(){
    double random = ((double) safe_rand())/safe_rand();
    return random;
}

#endif //SMARTSIM_CTEST_UTILS_H