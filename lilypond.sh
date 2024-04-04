#!/bin/bash

########
# Help #
########

Help() 
{
# list entries in bin
arg1list=()

dir=/home/evanmm3/bin

for entry in $dir/*
do 
    arg1list+=("${entry#$dir/}") 
done

# Display Help
echo "Performs Lilypond operations on a file."
echo
echo "Syntax: lilypond.sh [options] <operation> <file>"
echo
echo "Operations: "
printf -v joined '%s,' "${arg1list[@]}"
echo "${joined%", "}"
echo
echo "File: File to perform operation on"
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
            exit 1;;
   esac
done

shift $((OPTIND - 1))

if [ "$#" -ne 2 ]; then
    echo "Error: Missing arguments"
    exit 1
fi

operation=$1
file=$2

echo "Compiling $file with $operation..."

/home/evanmm3/bin/$operation $file 