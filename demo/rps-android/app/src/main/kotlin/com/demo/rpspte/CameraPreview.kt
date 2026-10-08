package com.demo.rpspte

import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.demo.rpspte.core.RgbaFrame
import java.util.concurrent.Executors

private const val TAG = "RpsCamera"

/** The preview and the analysis stream share one 4:3 aspect ratio, so the overlay lines up. */
private val ratio43 = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY

/**
 * Camera preview plus a frame stream for the classifier.
 *
 * Analysis frames are RGBA_8888 at about 640x480, newest-only (a slow frame is dropped,
 * never queued), and are NOT mirrored even for the front camera, which is how the
 * USB camera on the IQ9 behaves too. The preview is mirrored for the front camera so
 * it feels like a mirror to the player.
 */
@Composable
fun CameraPreview(
    front: Boolean,
    onFrame: (RgbaFrame) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentOnFrame by rememberUpdatedState(onFrame)

    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val executor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { executor.shutdown() } }

    DisposableEffect(front, lifecycleOwner) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder().setAspectRatioStrategy(ratio43).build()
                )
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(ratio43)
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(640, 480),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                            )
                        )
                        .build()
                )
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { useCase ->
                    useCase.setAnalyzer(executor) { image ->
                        try {
                            currentOnFrame(image.toRgbaFrame())
                        } catch (t: Throwable) {
                            Log.e(TAG, "analysis failed", t)
                        } finally {
                            image.close()
                        }
                    }
                }

            val selector = if (front) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            try {
                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, selector, preview, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "could not bind camera", e)
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            if (providerFuture.isDone) {
                try {
                    providerFuture.get().unbindAll()
                } catch (e: Exception) {
                    Log.w(TAG, "unbind failed", e)
                }
            }
        }
    }

    AndroidView(factory = { previewView }, modifier = modifier)
}

/** Copies the single RGBA plane out of the ImageProxy; the proxy is closed by the caller. */
private fun ImageProxy.toRgbaFrame(): RgbaFrame {
    val plane = planes[0]
    val buffer = plane.buffer
    buffer.rewind()
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return RgbaFrame(bytes, width, height, plane.rowStride, imageInfo.rotationDegrees)
}
