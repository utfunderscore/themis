plugins {
    `java-library`
    alias(libs.plugins.protobuf)
}

dependencies {
    implementation(libs.onnxruntime)
    implementation(libs.gson)
    implementation(libs.protobuf.java)

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

tasks.test {
    useJUnitPlatform()
    // Optional: -Pthemis.onnx.bundle=/path/to/exported/bundle enables OnnxModelTest.
    providers.gradleProperty("themis.onnx.bundle").orNull?.let {
        systemProperty("themis.onnx.bundle", it)
    }
}

protobuf {
    protoc {
        artifact = libs.protobuf.protoc.get().toString()
    }
}
