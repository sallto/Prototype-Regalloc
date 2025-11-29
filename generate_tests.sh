for filepath in $(find examples -type f -name "*.ir"); do
    python3 filecheck.py -k3 "${filepath}" --update
done
