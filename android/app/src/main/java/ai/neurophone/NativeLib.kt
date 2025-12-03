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
}
