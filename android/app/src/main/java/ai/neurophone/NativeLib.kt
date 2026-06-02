// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
package ai.neurophone

/**
 * Native library interface for the NeuroSymbolic system.
 * Connects Kotlin/Android to the Rust core via JNI.
 */
object NativeLib {

    init {
        System.loadLibrary("neurophone_android")
    }

    /**
     * Initialize the native system with optional JSON config
     */
    external fun init(configJson: String? = null): Boolean

    /**
     * Start the neural processing loop
     */
    external fun start(): Boolean

    /**
     * Stop the neural processing loop
     */
    external fun stop()

    /**
     * Process sensor data
     * @param sensorType Android sensor type constant
     * @param values Sensor values array
     * @param timestamp Event timestamp in nanoseconds
     * @param accuracy Sensor accuracy level
     */
    external fun processSensor(
        sensorType: Int,
        values: FloatArray,
        timestamp: Long,
        accuracy: Int
    ): Boolean

    /**
     * Query local LLM (Llama 3.2)
     */
    external fun queryLocal(message: String): String

    /**
     * Query Claude (cloud)
     */
    external fun queryClaude(message: String): String

    /**
     * Smart query - auto-selects local or cloud
     */
    external fun query(message: String, preferLocal: Boolean = true): String

    /**
     * Get current neural context as formatted string
     */
    external fun getNeuralContext(): String

    /**
     * Get system state as JSON
     */
    external fun getState(): String

    /**
     * Reset all neural components
     */
    external fun reset()

    /**
     * Check if system is running
     */
    external fun isRunning(): Boolean

    /**
     * Convenience wrapper used by the foreground service so the call site
     * is symmetric with the rest of the API even though the JNI signature
     * matches `processSensor` underneath.
     */
    fun pushSensorEvent(sensorType: String, timestampMs: Long, values: FloatArray): Boolean {
        val typeId = when (sensorType) {
            "accelerometer" -> 1
            "gyroscope"     -> 4
            "magnetometer"  -> 2
            "light"         -> 5
            "proximity"     -> 8
            else            -> 0
        }
        return processSensor(typeId, values, timestampMs * 1_000_000L, 3)
    }
}
