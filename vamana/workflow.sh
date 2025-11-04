# #!/bin/bash
# set -e 

# cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# if cmake --build build; then
#     ./build/fresh_vamana ../data/sift10k/sift10k_randomgraph.bin . .
# else
#     exit 1
# fi


#!/bin/bash
set -e

EXEC="./build/fresh_vamana"
ARGS="../data/sift10k/sift10k_randomgraph.bin . ."
REPORT_DIR="report"

usage() {
    echo "Usage: $0 [--build] [--run] [--ncu] [--nsys]"
    echo "  --build    Configure and compile project"
    echo "  --run      Run executable without profiling"
    echo "  --ncu      Run Nsight Compute profiler"
    echo "  --nsys     Run Nsight Systems profiler"
    echo ""
    echo "Examples:"
    echo "  $0 --build --run"
    echo "  $0 --run --ncu"
    echo "  $0 --nsys"
    exit 1
}

DO_BUILD=false
DO_RUN=false
DO_NCU=false
DO_NSYS=false

if [[ $# -eq 0 ]]; then
    usage
fi

for arg in "$@"; do
    case "$arg" in
        --build) DO_BUILD=true ;;
        --run)   DO_RUN=true ;;
        --ncu)   DO_NCU=true ;;
        --nsys)  DO_NSYS=true ;;
        -h|--help) usage ;;
        *) echo "[ERROR] Unknown option: $arg"; usage ;;
    esac
done

if $DO_BUILD; then
    echo "[INFO] Building project..."
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build build
fi

if [[ ! -f "$EXEC" ]]; then
    echo "[ERROR] Executable not found: $EXEC"
    echo "        Try running with --build first."
    exit 1
fi

mkdir -p "$REPORT_DIR"

last_report_num=$(find "$REPORT_DIR" -maxdepth 1 -type f -name "report_*.*" \
    | sed -E 's/.*report_([0-9]+)\..*/\1/' | sort -n | tail -1)

if [[ -z "$last_report_num" ]]; then
    next_report_num=1
else
    next_report_num=$((last_report_num + 1))
fi

if $DO_RUN; then
    echo "[INFO] Running executable..."
    "$EXEC" $ARGS
fi

if $DO_NCU; then
    if command -v ncu >/dev/null 2>&1; then
        echo "[INFO] Running Nsight Compute profiler..."
        ncu -o "$REPORT_DIR/report_${next_report_num}" "$EXEC" $ARGS
        echo "[INFO] Nsight Compute report saved to $REPORT_DIR/report_${next_report_num}.ncu-rep"
    else
        echo "[WARN] Nsight Compute (ncu) not found."
    fi
fi

if $DO_NSYS; then
    if command -v nsys >/dev/null 2>&1; then
        echo "[INFO] Running Nsight Systems profiler..."
        nsys profile -o "$REPORT_DIR/report_${next_report_num}" "$EXEC" $ARGS
        echo "[INFO] Nsight Systems report saved to $REPORT_DIR/report_${next_report_num}.qdrep"
    else
        echo "[WARN] Nsight Systems (nsys) not found."
    fi
fi

echo "[INFO] Done."
