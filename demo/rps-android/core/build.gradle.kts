// Copyright (c) 2026 EdgeImpulse Inc.
// SPDX-License-Identifier: BSD-3-Clause-Clear

plugins {
    kotlin("jvm")
}

// Pure Kotlin, no Android classes: preprocessing, the game engine and score handling.
// Keeping it free of android.* means it can be unit-tested on any JVM:
//
//     ./gradlew :core:test

// Same bytecode level as the app module, whichever JDK (17 or newer) runs Gradle.
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnit()
}
