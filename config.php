<?php

declare(strict_types=1);

/**
 * Minimal .env loader — no Composer dependency needed for a project this size.
 * Existing environment variables (e.g. set by the web server) always win.
 */
function loadEnvFile(string $path): void
{
    if (!is_file($path) || !is_readable($path)) {
        return;
    }

    $lines = file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    foreach ($lines as $line) {
        $line = trim($line);
        if ($line === '' || str_starts_with($line, '#') || !str_contains($line, '=')) {
            continue;
        }

        [$key, $value] = explode('=', $line, 2);
        $key = trim($key);
        $value = trim(trim($value), "\"'");

        if ($key !== '' && getenv($key) === false) {
            putenv("{$key}={$value}");
        }
    }
}

loadEnvFile(__DIR__ . '/.env');

return [
    'external_api' => [
        // Endpoint of the external Image Quality Validation API.
        'url' => getenv('EXTERNAL_API_URL') ?: '',
        // Optional bearer token, sent as "Authorization: Bearer <key>" if set.
        'api_key' => getenv('EXTERNAL_API_KEY') ?: '',
        // Field name the external API expects the file under.
        'field_name' => getenv('EXTERNAL_API_FIELD_NAME') ?: 'image',
        // Seconds before the outbound cURL request to the external API times out.
        'timeout' => (int) (getenv('EXTERNAL_API_TIMEOUT') ?: 15),
        'connect_timeout' => (int) (getenv('EXTERNAL_API_CONNECT_TIMEOUT') ?: 10),
    ],

    'upload' => [
        // multipart/form-data field name the mobile app must use.
        'field_name' => getenv('UPLOAD_FIELD_NAME') ?: 'image',
        // 10MB default cap.
        'max_size_bytes' => (int) (getenv('UPLOAD_MAX_SIZE_BYTES') ?: 10 * 1024 * 1024),
        'allowed_mime_types' => [
            'image/jpeg',
            'image/png',
            'image/webp',
        ],
    ],
];
