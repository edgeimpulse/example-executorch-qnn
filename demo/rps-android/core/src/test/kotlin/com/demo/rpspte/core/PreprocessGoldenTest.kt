package com.demo.rpspte.core

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Compares Preprocess.toNchw with the cv2 pipeline the IQ9 runs (rps.py).
 * The expected tensors come from tools/make_golden.py; regenerate with
 *
 *     python3 tools/make_golden.py
 *
 * cv2 resizes with fixed-point weights, so one level of 255 is the allowed difference.
 */
class PreprocessGoldenTest {

    private fun resource(name: String): ByteArray =
        checkNotNull(javaClass.getResourceAsStream("/golden/$name")) { "missing resource golden/$name" }
            .use { it.readBytes() }

    private data class Case(val name: String, val w: Int, val h: Int, val rot: Int, val roi: Int, val size: Int)

    private fun cases(): List<Case> =
        String(resource("index.txt")).lineSequence().filter { it.isNotBlank() }.map { line ->
            val p = line.trim().split(" ")
            Case(p[0], p[1].toInt(), p[2].toInt(), p[3].toInt(), p[4].toInt(), p[5].toInt())
        }.toList()

    private fun floats(bytes: ByteArray): FloatArray {
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    private fun check(case: Case, frame: RgbaFrame) {
        val expected = floats(resource("${case.name}.nchw"))
        val actual = Preprocess.toNchw(frame, case.size, case.roi)
        assertEquals(expected.size, actual.size, "${case.name}: tensor length")
        var worst = 0f
        for (i in expected.indices) worst = maxOf(worst, abs(expected[i] - actual[i]))
        assertTrue(worst * 255f <= 1.01f, "${case.name}: worst difference ${worst * 255f} levels of 255")
    }

    @Test
    fun matchesCv2OnAllGoldenCases() {
        val all = cases()
        assertTrue(all.isNotEmpty())
        for (c in all) {
            val frame = RgbaFrame(resource("${c.name}.rgba"), c.w, c.h, rotationDegrees = c.rot)
            check(c, frame)
        }
    }

    @Test
    fun rowStridePaddingIsIgnored() {
        // CameraX buffers can have padding at the end of each row; results must not change.
        for (c in cases()) {
            val tight = resource("${c.name}.rgba")
            val stride = c.w * 4 + 64
            val padded = ByteArray(stride * c.h) { 0x7F }
            for (y in 0 until c.h) System.arraycopy(tight, y * c.w * 4, padded, y * stride, c.w * 4)
            check(c, RgbaFrame(padded, c.w, c.h, rowStride = stride, rotationDegrees = c.rot))
        }
    }

    @Test
    fun outputLayoutIsNchwInZeroToOne() {
        val frame = RgbaFrame(ByteArray(8 * 8 * 4) { 0xFF.toByte() }, 8, 8)
        val out = Preprocess.toNchw(frame, 4)
        assertEquals(3 * 4 * 4, out.size)
        assertTrue(out.all { it == 1f })
    }

    @Test
    fun channelOrderIsRgbPlanar() {
        // A pure-red image must light up only the first plane.
        val px = ByteArray(8 * 8 * 4)
        for (i in 0 until 64) { px[i * 4] = 0xFF.toByte(); px[i * 4 + 3] = 0xFF.toByte() }
        val out = Preprocess.toNchw(RgbaFrame(px, 8, 8), 4)
        val plane = 16
        assertTrue((0 until plane).all { out[it] == 1f })
        assertTrue((plane until 3 * plane).all { out[it] == 0f })
    }

    @Test
    fun roiSideMatchesTheIq9Detector() {
        // pte_perception.roi_box(480) with ROI_FRACTION 0.6 is 288 pixels.
        val frame = RgbaFrame(ByteArray(640 * 480 * 4), 640, 480)
        assertEquals(288, Preprocess.roiSide(frame, 60))
        assertEquals(480, Preprocess.roiSide(frame, 100))
    }
}
