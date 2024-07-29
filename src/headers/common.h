#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// macro for better displacement of value
#define VALUE_TO_STRING(x) #x       // returns macro's name as string
#define VALUE(x) VALUE_TO_STRING(x) // returns macro's content as string

// macro to store time interval
#define TIMER_DEF     struct timeval temp_1, temp_2
#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)
#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)
#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

#endif