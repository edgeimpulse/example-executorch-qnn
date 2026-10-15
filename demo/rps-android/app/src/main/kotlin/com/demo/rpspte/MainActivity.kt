// Copyright (c) 2026 EdgeImpulse Inc.
// SPDX-License-Identifier: BSD-3-Clause-Clear

package com.demo.rpspte

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels

class MainActivity : ComponentActivity() {

    private val viewModel: RpsViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // This is a booth demo: the screen must not sleep between players.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent { RpsApp(viewModel) }
    }
}
