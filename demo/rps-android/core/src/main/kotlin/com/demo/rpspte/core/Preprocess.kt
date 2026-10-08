package com.demo.rpspte.core

/**
 * One camera frame as CameraX delivers it with OUTPUT_IMAGE_FORMAT_RGBA_8888.
 *
 * @param data            RGBA bytes (R, G, B, A per pixel), rows [rowStride] bytes apart
 * @param width,height    size of the buffer as delivered (NOT after rotation)
 * @param rotationDegrees clockwise rotation (0/90/180/270) that makes the image upright;
 *                        this is ImageInfo.rotationDegrees
 */
class RgbaFrame(
    val data: ByteArray,
    val width: Int,
    val height: Int,
    val rowStride: Int = width * 4,
    val rotationDegrees: Int = 0,
) {
    init {
        require(rotationDegrees in intArrayOf(0, 90, 180, 270)) { "rotation must be 0/90/180/270" }
        require(rowStride >= width * 4) { "rowStride smaller than a row" }
        require(data.size >= (height - 1) * rowStride + width * 4) { "buffer too small" }
    }

    val uprightWidth: Int get() = if (rotationDegrees % 180 == 0) width else height
    val uprightHeight: Int get() = if (rotationDegrees % 180 == 0) height else width
}

/**
 * Turns a camera frame into the tensor the .pte expects.
 *
 * This is the Android twin of Classifier.preprocess in rps.py / pte_perception.py:
 *
 *     rotate upright -> centre ROI square -> cv2.resize(INTER_LINEAR) -> RGB -> /255 -> NCHW
 *
 * It is pure Kotlin on purpose (no Bitmap, no android.*), so the same code runs in the
 * JVM unit tests, where it is compared with cv2 output (tools/make_golden.py).
 *
 * The bilinear sampling reproduces cv2.resize's default: source coordinate
 * (i + 0.5) * scale - 0.5, replicated borders, result rounded back to 8 bits before the
 * division by 255 (cv2 resizes uint8 images and returns uint8). cv2 uses fixed-point
 * weights, so results can differ from this by at most one level of 255.
 */
object Preprocess {

    /** Side, in pixels, of the square region that is classified. */
    fun roiSide(frame: RgbaFrame, roiPercent: Int): Int {
        require(roiPercent in 1..100) { "roiPercent must be 1..100" }
        return maxOf(1, minOf(frame.uprightWidth, frame.uprightHeight) * roiPercent / 100)
    }

    /**
     * @param inputSize  model input is [1, 3, inputSize, inputSize]
     * @param roiPercent side of the centred square as a percentage of the shorter side;
     *                   100 reproduces `rps.py` (centre-crop only)
     * @param mirror     flip left-right before cropping. The model was NOT trained on
     *                   mirrored frames, so this is off unless you have a reason.
     * @return NCHW float32 values in 0..1, ready for Tensor.fromBlob
     */
    fun toNchw(
        frame: RgbaFrame,
        inputSize: Int,
        roiPercent: Int = 100,
        mirror: Boolean = false,
    ): FloatArray {
        require(inputSize > 0)
        val uw = frame.uprightWidth
        val uh = frame.uprightHeight
        val side = roiSide(frame, roiPercent)
        val left = (uw - side) / 2
        val top = (uh - side) / 2

        // Per-axis sampling tables: two source taps and a weight for every output index.
        val x0 = IntArray(inputSize)
        val x1 = IntArray(inputSize)
        val wx = DoubleArray(inputSize)
        val y0 = IntArray(inputSize)
        val y1 = IntArray(inputSize)
        val wy = DoubleArray(inputSize)
        fillAxis(side, inputSize, x0, x1, wx)
        fillAxis(side, inputSize, y0, y1, wy)

        val plane = inputSize * inputSize
        val out = FloatArray(3 * plane)
        val buf = frame.data

        for (oy in 0 until inputSize) {
            val ya = top + y0[oy]
            val yb = top + y1[oy]
            val fy = wy[oy]
            for (ox in 0 until inputSize) {
                val xa = left + x0[ox]
                val xb = left + x1[ox]
                val fx = wx[ox]
                val o00 = offset(frame, xa, ya, mirror)
                val o01 = offset(frame, xb, ya, mirror)
                val o10 = offset(frame, xa, yb, mirror)
                val o11 = offset(frame, xb, yb, mirror)
                val idx = oy * inputSize + ox
                for (c in 0 until 3) {
                    val p00 = (buf[o00 + c].toInt() and 0xFF).toDouble()
                    val p01 = (buf[o01 + c].toInt() and 0xFF).toDouble()
                    val p10 = (buf[o10 + c].toInt() and 0xFF).toDouble()
                    val p11 = (buf[o11 + c].toInt() and 0xFF).toDouble()
                    val top2 = p00 * (1.0 - fx) + p01 * fx
                    val bot2 = p10 * (1.0 - fx) + p11 * fx
                    val v = top2 * (1.0 - fy) + bot2 * fy
                    val level = (v + 0.5).toInt().coerceIn(0, 255)
                    out[c * plane + idx] = level / 255f
                }
            }
        }
        return out
    }

    private fun fillAxis(side: Int, n: Int, a: IntArray, b: IntArray, w: DoubleArray) {
        val scale = side.toDouble() / n
        for (i in 0 until n) {
            val f = (i + 0.5) * scale - 0.5
            val i0 = Math.floor(f).toInt()
            w[i] = f - i0
            a[i] = i0.coerceIn(0, side - 1)
            b[i] = (i0 + 1).coerceIn(0, side - 1)
        }
    }

    /** Byte offset in the source buffer of the pixel at upright coordinates (ux, uy). */
    private fun offset(f: RgbaFrame, uxIn: Int, uy: Int, mirror: Boolean): Int {
        val ux = if (mirror) f.uprightWidth - 1 - uxIn else uxIn
        val sx: Int
        val sy: Int
        when (f.rotationDegrees) {
            0 -> { sx = ux; sy = uy }
            90 -> { sx = uy; sy = f.height - 1 - ux }
            180 -> { sx = f.width - 1 - ux; sy = f.height - 1 - uy }
            else -> { sx = f.width - 1 - uy; sy = ux }   // 270
        }
        return sy * f.rowStride + sx * 4
    }
}
