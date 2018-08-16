#ifndef __UTILS__
#define __UTILS__

#include <stdlib.h>
#include <stdio.h>

#define RUNTIME_ERROR(s)                    \
    {                                       \
        fprintf(stderr, "RUNTIME ERROR: "); \
        fprintf(stderr, s);                 \
        fprintf(stderr, "\n");              \
        exit(EXIT_FAILURE);                 \
    }

#define SIZE(a) (sizeof(a) / sizeof(a[0]))
#define PRINT_LIST(h, s, c, l)                  \
    if (c > 0) {                                \
        fprintf(h, s);                          \
        for(uint32_t i = 0; i < c; i++){        \
            fprintf(h, " "); fprintf(h, l[i]);  \
        }                                       \
        fprintf(h, "\n");                       \
    }

#endif