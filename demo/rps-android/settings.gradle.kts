// Copyright (c) 2026 EdgeImpulse Inc.
// SPDX-License-Identifier: BSD-3-Clause-Clear

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "rps-pte-android"
include(":app", ":core")
