plugins {
    id("io.papermc.paperweight.userdev") version "2.0.0-beta.24"
}

dependencies {
    paperweight.paperDevBundle("26.3.build.+")

    testImplementation(platform("org.junit:junit-bom:6.0.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    compileOnly("net.kyori:adventure-text-serializer-plain:5.2.0")
    implementation(project(":inference-onnx"))
}

java {
    toolchain.languageVersion.set(JavaLanguageVersion.of(25))
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(25)
}

tasks.test {
    useJUnitPlatform()
}
