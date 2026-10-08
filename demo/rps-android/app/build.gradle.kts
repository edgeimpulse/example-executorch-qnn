plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val executorchVersion = providers.gradleProperty("executorchVersion").get()

android {
    namespace = "com.demo.rpspte"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.demo.rpspte"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        // The Galaxy S26 is arm64. Keeping the APK to one ABI also keeps it small.
        ndk { abiFilters += "arm64-v8a" }

        // Shown in the app so the screen says which runtime produced the numbers.
        buildConfigField("String", "EXECUTORCH_VERSION", "\"$executorchVersion\"")
        buildConfigField("String", "BACKEND", "\"XNNPACK (CPU)\"")
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        jniLibs {
            // Other native libraries bundle their own libc++; take the first one.
            pickFirsts += "**/libc++_shared.so"
        }
    }

    // A .pte is a binary blob; never let aapt try to compress it.
    androidResources {
        noCompress += "pte"
    }
}

dependencies {
    implementation(project(":core"))

    // ExecuTorch runtime: Module / Tensor / EValue. The AAR on Maven Central is the
    // XNNPACK (CPU) build and brings fbjni and soloader along transitively.
    implementation("org.pytorch:executorch-android:$executorchVersion")

    // Camera
    val camerax = "1.4.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // UI
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.core:core-ktx:1.15.0")
}
