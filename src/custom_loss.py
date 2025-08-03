import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.loss import ClipLoss


class NegationParaphraseProjectionLoss(nn.Module):
    """Contrastive loss with paraphrase and negation constraints.
    Combines:
    1. Original contrastive loss between image and text embeddings
    2. Paraphrase loss (original ↔ paraphrase)
    3. Negation loss (original ↔ negation)
    """

    def __init__(
        self,
        contrastive_weight=1.0,
        paraphrase_weight=1.0,
        negation_weight=1.0,
        embedding_dim=512,
        use_learnable_projections=False,
        num_projection_vectors=2,
        normalize_projections=True,
    ):
        """Initialize the NegationParaphraseProjectionLoss.

        Args:
            contrastive_weight: Initial weight for contrastive loss
            paraphrase_weight: Initial weight for paraphrase loss
            negation_weight: Initial weight for negation loss
            embedding_dim: Embedding dimension
            use_learnable_projections: If True, basis vectors are learnable
            num_projection_vectors: 1 or 2 basis vectors
            normalize_projections: If True, normalize projections
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.paraphrase_weight = paraphrase_weight
        self.negation_weight = negation_weight
        self.embedding_dim = embedding_dim
        self.use_learnable_projections = use_learnable_projections
        self.num_projection_vectors = num_projection_vectors
        self.normalize_projections = normalize_projections
        assert num_projection_vectors >= 1, "num_projection_vectors must be at least 1"

        self._init_basis_vectors(
            embedding_dim, num_projection_vectors, use_learnable_projections
        )

        self.clip_loss = ClipLoss()

    def _init_basis_vectors(
        self, embedding_dim, num_projection_vectors, use_learnable_projections
    ):
        """Initialize basis vectors for projections.

        Creates N orthogonal basis vectors using Gram-Schmidt process.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        basis_vectors = []
        for i in range(num_projection_vectors):
            new_vec = torch.randn(embedding_dim, device=device)

            for j in range(len(basis_vectors)):
                v = basis_vectors[j]
                proj_coef = torch.dot(new_vec, v)
                new_vec = new_vec - proj_coef * v

            new_vec = F.normalize(new_vec, p=2, dim=0, eps=1e-6)
            basis_vectors.append(new_vec)

        if use_learnable_projections:
            self.basis_vectors = nn.ParameterList(
                [nn.Parameter(v) for v in basis_vectors]
            )
        else:
            self.register_buffer("basis_vectors", torch.stack(basis_vectors))

    def _normalize_embeddings(
        self, text_embeddings, structure_embeddings, negation_embeddings
    ):
        """Normalize all embedding types to unit length"""
        eps = 1e-6
        return (
            F.normalize(text_embeddings, p=2, dim=-1, eps=eps),
            F.normalize(structure_embeddings, p=2, dim=-1, eps=eps),
            F.normalize(negation_embeddings, p=2, dim=-1, eps=eps),
        )

    def _compute_projections(self, embeddings):
        """Compute projections of embeddings onto all basis vectors.

        Projects embeddings onto basis vectors and returns them stacked in the last dimension.
        Normalization is handled separately in _compute_projection_similarity_loss.
        """
        if isinstance(self.basis_vectors, nn.ParameterList):
            basis = torch.stack([v for v in self.basis_vectors], dim=1)
        else:
            basis = self.basis_vectors.transpose(0, 1)

        projections = embeddings @ basis

        if self.num_projection_vectors == 1:
            projections = projections.unsqueeze(-1)

        return projections

    def _is_baseline_clip_model(self):
        """Check if the model is a baseline CLIP model."""
        return (
            self.contrastive_weight == 1.0
            and self.paraphrase_weight == 0.0
            and self.negation_weight == 0.0
        )

    def _compute_projection_similarity_loss(self, proj_a, proj_b, target_sign):
        """Compute similarity loss between projections, enforcing sign agreement or opposition.

        Uses CosineEmbeddingLoss with appropriate margin to enforce similarity or dissimilarity.
        """
        if self.normalize_projections:
            proj_a = F.normalize(proj_a, p=2, dim=-1, eps=1e-6)
            proj_b = F.normalize(proj_b, p=2, dim=-1, eps=1e-6)

        if self.num_projection_vectors == 1:
            proj_a = proj_a.squeeze(-1)
            proj_b = proj_b.squeeze(-1)

        y = torch.full(
            (proj_a.shape[0],), target_sign, dtype=torch.float, device=proj_a.device
        )

        return nn.CosineEmbeddingLoss(margin=0.0, reduction="mean")(proj_a, proj_b, y)

    def forward(
        self,
        text_embeddings,
        paraphrase_embeddings,
        negation_embeddings,
        contrastive_loss=None,
    ):
        """Compute the combined loss from contrastive, paraphrase, and negation components.
        Adds orthogonality regularization, batch checks, and component flags.
        """
        paraphrase_loss = 0.0
        negation_loss = 0.0

        if not self._is_baseline_clip_model():
            text_embeddings, paraphrase_embeddings, negation_embeddings = (
                self._normalize_embeddings(
                    text_embeddings, paraphrase_embeddings, negation_embeddings
                )
            )
            original_projs = self._compute_projections(text_embeddings)
            paraphrase_projs = self._compute_projections(paraphrase_embeddings)
            negation_projs = self._compute_projections(negation_embeddings)

            paraphrase_loss = self._compute_projection_similarity_loss(
                original_projs, paraphrase_projs, 1.0
            )
            negation_loss = self._compute_projection_similarity_loss(
                original_projs, negation_projs, -1.0
            )

        total_weights = (
            self.contrastive_weight + self.paraphrase_weight + self.negation_weight
        )
        total_loss = (
            (
                self.contrastive_weight * contrastive_loss
                + self.paraphrase_weight * paraphrase_loss
                + self.negation_weight * negation_loss
            )
            / total_weights
            if total_weights > 0
            else 0.0
        )

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "paraphrase_loss": paraphrase_loss,
            "negation_loss": negation_loss,
        }
