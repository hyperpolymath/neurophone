plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.neurophone"
    compileSdk = 34

    defaultConfig {
        applicationId = "ai.neurophone"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // Supported ABIs - arm64-v8a for Oppo Reno 13 (Dimensity 8350)
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }

        externalNativeBuild {
            cmake {
                cppFlags += ""
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    // Native library location
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Coroutines for async operations
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // JSON parsing
    implementation("org.json:json:20231013")

    // Lifecycle
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // ---------------------------------------------------------------------
    // Gossamer Android webview shell (epic #83, RFC #97, sub-issue #109).
    //
    // NeurophoneMainActivity extends io.gossamer.GossamerActivity. As of
    // sub-PR #3 (scaffold only) the Gossamer Android library is NOT published
    // to any Maven repo — upstream (hyperpolymath/gossamer) ships it as a raw
    // Java source tree (android/src/main/java/io/gossamer/*.java) plus a
    // prebuilt libgossamer.so. So there is nothing resolvable to depend on yet
    // and this is left commented to keep the scaffold buildable in isolation.
    //
    // TODO(#83 sub-PR #4): consume the gossamer Android library once available,
    // e.g. as a vendored module, a local AAR, or a published coordinate:
    //
    //   implementation("io.gossamer:gossamer-android:<version>")
    //   // and drop libgossamer.so into app/src/main/jniLibs/<abi>/
    // ---------------------------------------------------------------------

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
