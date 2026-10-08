#!/usr/bin/env bash
# Read-only: where the QNN runtime libraries, headers and Hexagon skel libraries
# are on this board. Use it to set ADSP_LIBRARY_PATH on boards other than the IQ-9075.
echo "== QNN backend/system libs (main + stub + prepare)"
find /usr /opt -type f \( -name 'libQnnHtp.so' -o -name 'libQnnHtpPrepare.so' -o -name 'libQnnSystem.so' \
   -o -name 'libQnnHtpV73Stub.so' -o -name 'libQnnHtpV73CalculatorStub.so' \) 2>/dev/null
echo "== all libQnn* in /usr/lib"
ls -l /usr/lib/libQnn* /usr/lib/aarch64-linux-gnu/libQnn* 2>/dev/null
echo "== headers"
ls /usr/include/QNN | head -30
echo "== dsp skel dirs (SA8775P = QCS9100 family, Hexagon v73)"
ls -l /usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp/
echo "== user / groups (fastrpc access)"; id
echo "== compilers"; g++ --version | head -1; python3 --version; nproc; free -g | head -2; df -h $HOME | tail -1
