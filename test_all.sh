failed=0
for file in $(find examples -type f -name "*.ir"); do
    python3 filecheck.py "$file"
    if [ $? -ne 0 ]; then
        failed=1
    fi
done

if [ $failed -eq 1 ]; then
    echo "Some checks failed"
    exit 1
else
    echo "all checks passed"
fi
