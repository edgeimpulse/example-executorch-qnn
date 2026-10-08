package com.demo.rpspte

import android.content.Context
import java.io.File
import java.io.IOException
import java.security.MessageDigest

/**
 * Finds the model. Module.load() needs a real file path, so a bundled asset is copied
 * into the app's private storage first.
 *
 * Lookup order:
 *   1. /sdcard/Android/data/<package>/files/model.pte   (adb push, no rebuild needed)
 *   2. app/src/main/assets/model.pte                    (bundled in the APK)
 */
object ModelFiles {

    data class Resolved(val model: File, val labels: File, val source: String)

    fun resolve(context: Context): Resolved? {
        val external = context.getExternalFilesDir(null)
        val extModel = external?.let { File(it, MODEL) }
        if (extModel != null && extModel.isFile) {
            val extLabels = File(external, LABELS)
            val labels = if (extLabels.isFile) extLabels else copyAsset(context, LABELS) ?: return null
            return Resolved(extModel, labels, "adb push")
        }
        val model = copyAsset(context, MODEL) ?: return null
        val labels = copyAsset(context, LABELS) ?: return null
        return Resolved(model, labels, "bundled asset")
    }

    /** First 8 hex characters of the SHA-256, so the screen can show "same model" at a glance. */
    fun fingerprint(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                digest.update(buf, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }.take(8)
    }

    /** Always re-copied (a couple of MB), so replacing the asset takes effect on the next install. */
    private fun copyAsset(context: Context, name: String): File? = try {
        val out = File(context.filesDir, name)
        context.assets.open(name).use { input -> out.outputStream().use { input.copyTo(it) } }
        out
    } catch (e: IOException) {
        null
    }

    const val MODEL = "model.pte"
    const val LABELS = "labels.txt"
}
