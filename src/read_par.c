#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "main.h"

int read_int(char *ini_file, char *par_name, int *parameter)
{
    char line[MAX_LINE],
         ini_par_name[MAX_LINE],
         par_separator[MAX_LINE],
         par_value[MAX_LINE];

    FILE *par_file_ptr;

    if ((par_file_ptr = fopen(ini_file, "r")) == NULL)
    {
        printf("Cannot open parameter file '%s'.\n", ini_file);
        exit(-1);
    }

    int linenum = 0;
    *parameter = NO_PAR_CONST;

    while(fgets(line, MAX_LINE, par_file_ptr) != NULL)
    {
        linenum++;
        if (line[0] == '#' || line[0] == '\r')
            continue;

        if (sscanf(line, "%s %s %s", ini_par_name, par_separator, par_value) \
            != 3)
        {
            printf("Syntax error: line %d for parameter %s\n", linenum,
                   par_name);
            exit(-1);
        }

        if (strcmp(par_name, ini_par_name) == 0)
            *parameter =  atoi(par_value);
    }

    /*
     * If parameter wasn't updated, the parameter was not found in the file so
     * look up a default value to use
     */
    if (*parameter == NO_PAR_CONST)
        get_int_CL(par_name, parameter);

    if (fclose(par_file_ptr) != 0)
    {
            printf("Parameter file '%s' could not be closed.\n", ini_file);
            exit(-1);
    }

    return 0;
}

int get_int_CL(char *par_name, int *parameter)
{
    char par_input[MAX_LINE];

    printf("INT: %s\n", par_name);
    scanf("%s", par_input);
    *parameter = atoi(par_input);

    return 0;
}

int read_string(char *ini_file, char *par_name, char *parameter)
{
    char line[MAX_LINE], ini_par_name[MAX_LINE], par_separator[MAX_LINE];
    char par_value[MAX_LINE];

    FILE *par_file;

    if ((par_file = fopen(ini_file, "r")) == NULL)
    {
        printf("Cannot open parameter file '%s'.\n", ini_file);
        exit(-1);
    }

    int linenum = 0;
    parameter[0] = STRING_NO_PAR_CONST;

    while(fgets(line, MAX_LINE, par_file) != NULL)
    {
        linenum++;

        if (line[0] == '#' || line[0] == '\r')
        {
            continue;
        }

        if (sscanf(line, "%s %s %s", ini_par_name, par_separator, par_value) \
            != 3)
        {
            printf("Syntax error, line %d for parameter %s\n", linenum,
                par_name);
            exit(-1);
        }

        /*
        * Use strcmp to compare the difference between two strings. If it's
        * the same parameter, then strcmp will return 0.
        */
        if (strcmp(par_name, ini_par_name) == 0)
        {
            strcpy(parameter, par_value);
        }
    }

    /*
    * If parameter wasn't updated, the parameter was not found. Return an
    * error.
    */
    if (parameter[0] == STRING_NO_PAR_CONST)
        get_string_CL(par_name, parameter);

    if (fclose(par_file) != 0)
    {
        printf("File could not be closed.\n");
        exit(-1);
    }

    return 0;
}

int get_string_CL(char *par_name, char *parameter)
{
    char par_input[MAX_LINE];

    printf("INT: %s\n", par_name);
    scanf("%s", par_input);
    strcpy(parameter, par_input);

    return 0;
}
