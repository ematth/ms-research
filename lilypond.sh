#!/bin/bash

########
# Help #
########

Help() 
{
    # Display Help
    echo "Performs Lilypond operations on a file."
    echo
    echo "Syntax: lilypond.sh (1) (2)"
    echo "arguments:"
    echo "(1)   lilypond, "
    echo "(2)     Print this Help."
    echo
    echo "options:"
    echo "-h    Print this help menu."
    echo
}

########
# Main #
########

while getopts ":h" option; do
   case $option in
        h) # display Help
            Help
            exit;;
        \?) # Invalid option
            echo "Error: Invalid option"
            exit;;
   esac
done


echo "Compiling $1 with $2..."
# The magic line :)
# ../bin/$1 $2;