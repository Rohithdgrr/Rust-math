# MathVerse Workspace Validation Script
# Run this script to validate the workspace is production-ready

param(
    [switch]$SkipBuild,
    [switch]$SkipTests,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"
$workspaceRoot = Split-Path -Parent $PSScriptRoot
Set-Location $workspaceRoot

Write-Host "🔍 MathVerse Workspace Validation" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Counter for issues
$issueCount = 0
$warningCount = 0

# 1. Check all crates are in workspace members
Write-Host "✓ Checking workspace membership..." -ForegroundColor Yellow
$cratesDir = Join-Path $workspaceRoot "crates"
$crateNames = Get-ChildItem $cratesDir -Directory | ForEach-Object { $_.Name }
$cargoToml = Get-Content (Join-Path $workspaceRoot "Cargo.toml") -Raw

foreach ($crate in $crateNames) {
    if ($cargoToml -notmatch [regex]::Escape("crates/$crate")) {
        Write-Host "  ❌ $crate is NOT in workspace members!" -ForegroundColor Red
        $issueCount++
    } elseif ($Verbose) {
        Write-Host "  ✓ $crate" -ForegroundColor Green
    }
}

if ($issueCount -eq 0) {
    Write-Host "  ✅ All $($crateNames.Count) crates are workspace members`n" -ForegroundColor Green
}

# 2. Check repository URLs
Write-Host "✓ Checking repository URLs..." -ForegroundColor Yellow
$correctUrl = "https://github.com/Rohithdgrr/Rust-math"
foreach ($crate in $crateNames) {
    $crateToml = Join-Path $cratesDir $crate "Cargo.toml"
    if (Test-Path $crateToml) {
        $content = Get-Content $crateToml -Raw
        if ($content -match 'repository\s*=\s*"([^"]+)"') {
            $url = $matches[1]
            if ($url -ne $correctUrl) {
                Write-Host "  ❌ $crate has wrong URL: $url" -ForegroundColor Red
                $issueCount++
            } elseif ($Verbose) {
                Write-Host "  ✓ $crate" -ForegroundColor Green
            }
        }
    }
}
Write-Host "  ✅ Repository URL check complete`n" -ForegroundColor Green

# 3. Check for README files
Write-Host "✓ Checking README files..." -ForegroundColor Yellow
$missingReadmes = @()
foreach ($crate in $crateNames) {
    $readmePath = Join-Path $cratesDir $crate "README.md"
    if (-not (Test-Path $readmePath)) {
        $missingReadmes += $crate
        $warningCount++
    } elseif ($Verbose) {
        Write-Host "  ✓ $crate" -ForegroundColor Green
    }
}

if ($missingReadmes.Count -gt 0) {
    Write-Host "  ⚠️  Missing READMEs: $($missingReadmes -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host "  ✅ All crates have READMEs`n" -ForegroundColor Green
}

# 4. Check mathverse-core dependency
Write-Host "✓ Checking mathverse-core dependencies..." -ForegroundColor Yellow
$shouldDependOnCore = @(
    'mathverse-complex', 'mathverse-signal', 'mathverse-graph', 
    'mathverse-statistics', 'mathverse-algebra', 'mathverse-calculus',
    'mathverse-probability', 'mathverse-numerical', 'mathverse-finance',
    'mathverse-physics', 'mathverse-symbolic', 'mathverse-units'
)

foreach ($crate in $shouldDependOnCore) {
    $crateToml = Join-Path $cratesDir $crate "Cargo.toml"
    if (Test-Path $crateToml) {
        $content = Get-Content $crateToml -Raw
        if ($content -notmatch 'mathverse-core') {
            Write-Host "  ❌ $crate doesn't depend on mathverse-core!" -ForegroundColor Red
            $issueCount++
        } elseif ($Verbose) {
            Write-Host "  ✓ $crate" -ForegroundColor Green
        }
    }
}
Write-Host "  ✅ Core dependency check complete`n" -ForegroundColor Green

# 5. Check for workspace lints adoption
Write-Host "✓ Checking workspace lints adoption..." -ForegroundColor Yellow
$missingLints = @()
foreach ($crate in $crateNames) {
    $crateToml = Join-Path $cratesDir $crate "Cargo.toml"
    if (Test-Path $crateToml) {
        $content = Get-Content $crateToml -Raw
        if ($content -notmatch '\[lints\][\s\S]*?workspace\s*=\s*true') {
            $missingLints += $crate
            $warningCount++
        } elseif ($Verbose) {
            Write-Host "  ✓ $crate" -ForegroundColor Green
        }
    }
}

if ($missingLints.Count -gt 0) {
    Write-Host "  ⚠️  Missing workspace lints: $($missingLints -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host "  ✅ All crates adopt workspace lints`n" -ForegroundColor Green
}

# 6. Check for unsafe code
Write-Host "✓ Scanning for unsafe code..." -ForegroundColor Yellow
$unsafeFiles = @()
Get-ChildItem -Path $cratesDir -Recurse -Include "*.rs" | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if ($content -match '\bunsafe\b') {
        $relativePath = $_.FullName.Replace($workspaceRoot, "").TrimStart('\')
        $unsafeFiles += $relativePath
    }
}

if ($unsafeFiles.Count -gt 0) {
    Write-Host "  ⚠️  Found unsafe code in:" -ForegroundColor Yellow
    $unsafeFiles | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
    $warningCount++
} else {
    Write-Host "  ✅ No unsafe code found`n" -ForegroundColor Green
}

# 7. Check image crate dependency
Write-Host "✓ Checking mathverse-image dependencies..." -ForegroundColor Yellow
$imageToml = Join-Path $cratesDir "mathverse-image" "Cargo.toml"
if (Test-Path $imageToml) {
    $content = Get-Content $imageToml -Raw
    if ($content -match 'image\s*=\s*\{[^}]*default-features\s*=\s*false[^}]*\}') {
        Write-Host "  ✅ mathverse-image uses minimal image features`n" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  mathverse-image may pull in unnecessary dependencies" -ForegroundColor Yellow
        $warningCount++
    }
}

# 8. Build checks (if not skipped)
if (-not $SkipBuild) {
    Write-Host "✓ Running cargo check..." -ForegroundColor Yellow
    try {
        $checkOutput = cargo check --workspace 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ cargo check passed`n" -ForegroundColor Green
        } else {
            Write-Host "  ❌ cargo check FAILED:" -ForegroundColor Red
            Write-Host $checkOutput -ForegroundColor Red
            $issueCount++
        }
    } catch {
        Write-Host "  ⚠️  cargo not available, skipping build check" -ForegroundColor Yellow
        $warningCount++
    }

    # Clippy check
    Write-Host "✓ Running cargo clippy..." -ForegroundColor Yellow
    try {
        $clippyOutput = cargo clippy --workspace --all-targets -- -D warnings 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ cargo clippy passed`n" -ForegroundColor Green
        } else {
            Write-Host "  ❌ cargo clippy found issues:" -ForegroundColor Red
            Write-Host $clippyOutput -ForegroundColor Red
            $issueCount++
        }
    } catch {
        Write-Host "  ⚠️  cargo not available, skipping clippy check" -ForegroundColor Yellow
        $warningCount++
    }
}

# 9. Test checks (if not skipped)
if (-not $SkipTests) {
    Write-Host "✓ Running cargo test..." -ForegroundColor Yellow
    try {
        $testOutput = cargo test --workspace 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ All tests passed`n" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Some tests FAILED:" -ForegroundColor Red
            Write-Host $testOutput -ForegroundColor Red
            $issueCount++
        }
    } catch {
        Write-Host "  ⚠️  cargo not available, skipping test check" -ForegroundColor Yellow
        $warningCount++
    }
}

# 10. Documentation coverage (quick scan)
Write-Host "✓ Checking documentation coverage..." -ForegroundColor Yellow
$undocumentedCrates = @()
foreach ($crate in $crateNames) {
    $srcDir = Join-Path $cratesDir $crate "src"
    if (Test-Path $srcDir) {
        $hasDocComments = $false
        Get-ChildItem -Path $srcDir -Recurse -Filter "*.rs" | ForEach-Object {
            $content = Get-Content $_.FullName -Raw
            if ($content -match '///|//!') {
                $hasDocComments = $true
            }
        }
        if (-not $hasDocComments) {
            $undocumentedCrates += $crate
        }
    }
}

if ($undocumentedCrates.Count -gt 0) {
    Write-Host "  ⚠️  Crates with no doc comments ($($undocumentedCrates.Count)):" -ForegroundColor Yellow
    $undocumentedCrates | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
    $warningCount++
} else {
    Write-Host "  ✅ All crates have some documentation`n" -ForegroundColor Green
}

# Summary
Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "📊 Validation Summary" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

if ($issueCount -eq 0 -and $warningCount -eq 0) {
    Write-Host "🎉 PERFECT! No issues or warnings found." -ForegroundColor Green
    Write-Host "   The workspace is production-ready!`n" -ForegroundColor Green
    exit 0
} elseif ($issueCount -eq 0) {
    Write-Host "✅ GOOD! No critical issues found." -ForegroundColor Green
    Write-Host "⚠️  $warningCount warning(s) to address.`n" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "❌ ISSUES FOUND!" -ForegroundColor Red
    Write-Host "   $issueCount critical issue(s)" -ForegroundColor Red
    Write-Host "   $warningCount warning(s)`n" -ForegroundColor Yellow
    Write-Host "Please fix critical issues before publishing.`n" -ForegroundColor Red
    exit 1
}
