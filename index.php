<?php

declare(strict_types=1);

header('Content-Type: application/json');

require_once __DIR__ . '/ImageValidator.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    header('Allow: POST');
    echo json_encode(['status' => 'error', 'message' => 'Method not allowed. Use POST.']);
    exit;
}

$config = require __DIR__ . '/config.php';

$validator = new ImageValidator($config);
$result = $validator->handleUpload($_FILES[$config['upload']['field_name']] ?? null);

http_response_code($result['httpStatus']);
echo json_encode($result['body']);
