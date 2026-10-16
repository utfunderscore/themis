plugins {
    application
}

dependencies {
    implementation(project(":inference-onnx"))
}

application {
    mainClass.set("lol.utf.themis.cli.ModerationCli")
}

tasks.named<JavaExec>("run") {
    standardInput = System.`in`
}
